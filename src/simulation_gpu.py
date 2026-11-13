"""
YAXCHE - Motor de Simulacion Masivamente Paralelo en GPU (CuPy + CUDA RawKernel)
Disenado para los nodos NVIDIA L40S y V100 del Cluster Kabre (CeNAT).
Opera en float32 para aprovechar la tasa 43x FP32/FP64 de la arquitectura L40S,
con mapeo coalescente MAP_ROWS en sectores de 32 bytes y doble buffering en VRAM.
"""

from __future__ import annotations

import math
import time
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


CUDA_KERNEL_SOURCE = r"""
// Generador stateless SplitMix32 inline para transiciones estocasticas en GPU
__device__ __forceinline__ float rand_u01(
    unsigned int seed, int step, int r, int c, int k
) {
    unsigned int x = seed;
    x ^= (unsigned int)step * 0x9E3779B9u;
    x ^= (unsigned int)r    * 0x85EBCA6Bu;
    x ^= (unsigned int)c    * 0xC2B2AE35u;
    x ^= (unsigned int)k    * 0x27D4EB2Fu;
    x = (x ^ (x >> 16)) * 0x7FEB352Du;
    x = (x ^ (x >> 15)) * 0x846CA68Bu;
    x = x ^ (x >> 16);
    return (float)x * 2.3283064365386963e-10f; // 1.0f / 2^32
}

extern "C" __global__
void yaxche_fire_step(
    const int* __restrict__ state_in,
    int* __restrict__ state_out,
    const int* __restrict__ timer_in,
    int* __restrict__ timer_out,
    float* __restrict__ toa,
    const float* __restrict__ elevation,
    const float* __restrict__ fuel,
    const unsigned char* __restrict__ mask_gt,
    const float* __restrict__ wind_u,
    const float* __restrict__ wind_v,
    const float cell_size,
    const float next_time_hours,
    const int burn_duration_steps,
    const int step_idx,
    const unsigned int seed,
    const int rows,
    const int cols
) {
#ifdef MAP_ROWS
    // Coalescente: hilos consecutivos en threadIdx.x leen columnas contiguas (c)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
#else
    // No coalescente (strided): para medicion comparativa de ancho de banda
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
#endif

    if (r <= 0 || r >= rows - 1 || c <= 0 || c >= cols - 1) {
        return;
    }

    const int idx = r * cols + c;
    const int st = state_in[idx];

    if (st == 2) { // STATE_BURNED
        state_out[idx] = 2;
        timer_out[idx] = timer_in[idx];
        return;
    }

    if (st == 1) { // STATE_BURNING
        const int t_next = timer_in[idx] + 1;
        timer_out[idx] = t_next;
        state_out[idx] = (t_next >= burn_duration_steps) ? 2 : 1;
        return;
    }

    // STATE_UNBURNED
    if (mask_gt[idx] == 0 || fuel[idx] <= 0.08f) {
        state_out[idx] = 0;
        timer_out[idx] = 0;
        return;
    }

    const float h_target = elevation[idx];
    const float f_target = fuel[idx];
    const float inv_sqrt2 = 0.70710678f;

    const int drs[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    const int dcs[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const float inv_dists[8] = {
        inv_sqrt2, 1.0f, inv_sqrt2,
        1.0f,            1.0f,
        inv_sqrt2, 1.0f, inv_sqrt2
    };

    bool ignited = false;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int nr = r + drs[k];
        const int nc = c + dcs[k];
        const int nidx = nr * cols + nc;

        if (state_in[nidx] == 1 && mask_gt[nidx] != 0) {
            const float inv_d = inv_dists[k];
            float dh_norm = (h_target - elevation[nidx]) * inv_d / cell_size;
            dh_norm = fminf(1.0f, fmaxf(-1.0f, dh_norm));

            const float dir_x = -((float)dcs[k]) * inv_d;
            const float dir_y =  ((float)drs[k]) * inv_d;
            const float wind_dot = wind_u[nidx] * dir_x + wind_v[nidx] * dir_y;

            const float scale_factor = fminf(1.0f, fmaxf(0.15f, 100.0f / cell_size));
            float prob = (0.18f * scale_factor) * f_target * expf(2.5f * dh_norm) * expf(0.35f * wind_dot);
            prob = fminf(0.95f, fmaxf(0.0f, prob));

            if (rand_u01(seed, step_idx, r, c, k) < prob) {
                ignited = true;
                break;
            }
        }
    }

    if (ignited) {
        state_out[idx] = 1;
        timer_out[idx] = 0;
        toa[idx] = next_time_hours;
    } else {
        state_out[idx] = 0;
        timer_out[idx] = 0;
    }
}
"""


def _require_cupy() -> None:
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy no esta disponible en este entorno. Ejecute este modulo en un nodo GPU de Kabre (nukwa-l40s / nukwa-v100)."
        )


def compile_kernel(coalesced: bool = True):
    _require_cupy()
    opts = ("-DMAP_ROWS",) if coalesced else ()
    return cp.RawKernel(CUDA_KERNEL_SOURCE, "yaxche_fire_step", options=opts)


def run_gpu_simulation(
    state_init: np.ndarray,
    elevation: np.ndarray,
    fuel: np.ndarray,
    mask_guatemala: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cell_size: float,
    steps: int = 48,
    dt_hours: float = 1.0,
    burn_duration_steps: int = 5,
    seed: int = 42,
    block_dim: tuple[int, int] = (16, 16),
    coalesced: bool = True,
) -> dict:
    """
    Ejecuta N pasos completos en VRAM con doble buffering y retorna resultados y tiempos
    separados (transferencia H2D, kernel puro con CUDA Events, transferencia D2H, total).
    """
    _require_cupy()
    rows, cols = state_init.shape
    kernel = compile_kernel(coalesced=coalesced)

    if coalesced:
        grid_dim = (math.ceil(cols / block_dim[0]), math.ceil(rows / block_dim[1]))
    else:
        grid_dim = (math.ceil(rows / block_dim[0]), math.ceil(cols / block_dim[1]))

    device = cp.cuda.Device()
    device.synchronize()
    t_total_0 = time.perf_counter()

    # Transferencia Host -> Device (una sola vez)
    t_h2d_0 = time.perf_counter()
    d_state_a = cp.asarray(state_init, dtype=cp.int32)
    d_state_b = d_state_a.copy()
    d_timer_a = cp.zeros((rows, cols), dtype=cp.int32)
    d_timer_b = cp.zeros((rows, cols), dtype=cp.int32)
    d_toa = cp.full((rows, cols), cp.nan, dtype=cp.float32)
    d_toa[d_state_a == 1] = 0.0

    d_elev = cp.asarray(elevation, dtype=cp.float32)
    d_fuel = cp.asarray(fuel, dtype=cp.float32)
    d_mask = cp.asarray(mask_guatemala, dtype=cp.uint8)
    d_wu = cp.asarray(wind_u, dtype=cp.float32)
    d_wv = cp.asarray(wind_v, dtype=cp.float32)
    device.synchronize()
    h2d_s = time.perf_counter() - t_h2d_0

    # Medicion pura del kernel en Device mediante CUDA Events
    ev_start = cp.cuda.Event()
    ev_stop = cp.cuda.Event()
    ev_start.record()

    for s in range(1, steps + 1):
        kernel(
            grid_dim,
            block_dim,
            (
                d_state_a,
                d_state_b,
                d_timer_a,
                d_timer_b,
                d_toa,
                d_elev,
                d_fuel,
                d_mask,
                d_wu,
                d_wv,
                np.float32(cell_size),
                np.float32(s * dt_hours),
                np.int32(burn_duration_steps),
                np.int32(s),
                np.uint32(seed),
                np.int32(rows),
                np.int32(cols),
            ),
        )
        d_state_a, d_state_b = d_state_b, d_state_a
        d_timer_a, d_timer_b = d_timer_b, d_timer_a

    ev_stop.record()
    ev_stop.synchronize()
    kernel_s = cp.cuda.get_elapsed_time(ev_start, ev_stop) / 1000.0

    # Transferencia Device -> Host (una sola vez al final)
    t_d2h_0 = time.perf_counter()
    final_state = cp.asnumpy(d_state_a)
    final_toa = cp.asnumpy(d_toa)
    device.synchronize()
    d2h_s = time.perf_counter() - t_d2h_0

    total_s = time.perf_counter() - t_total_0

    return {
        "state": final_state,
        "toa": final_toa,
        "h2d_s": h2d_s,
        "kernel_s": kernel_s,
        "d2h_s": d2h_s,
        "total_s": total_s,
    }

