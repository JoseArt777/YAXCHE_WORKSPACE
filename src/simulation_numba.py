"""
YAXCHE - Motor de Simulacion Paralelo en CPU (Numba JIT + prange)
Implementa el automata celular 2D con formulacion gather (pull) libre de
condiciones de carrera, doble buffering y PRNG stateless por celda.
"""

from __future__ import annotations

import math
import numba
import numpy as np

from src.simulation import STATE_BURNED, STATE_BURNING, STATE_UNBURNED


@numba.njit(inline="always")
def _rand_u01(seed: np.uint32, step: np.int32, r: np.int32, c: np.int32, k: np.int32) -> float:
    """Generador pseudoaleatorio stateless (hash SplitMix32) en [0, 1)."""
    mask32 = numba.uint64(0xFFFFFFFF)
    x = numba.uint64(seed) & mask32
    x = (x ^ ((numba.uint64(step) * numba.uint64(0x9E3779B9)) & mask32)) & mask32
    x = (x ^ ((numba.uint64(r) * numba.uint64(0x85EBCA6B)) & mask32)) & mask32
    x = (x ^ ((numba.uint64(c) * numba.uint64(0xC2B2AE35)) & mask32)) & mask32
    x = (x ^ ((numba.uint64(k) * numba.uint64(0x27D4EB2F)) & mask32)) & mask32
    x = ((x ^ (x >> numba.uint64(16))) * numba.uint64(0x7FEB352D)) & mask32
    x = ((x ^ (x >> numba.uint64(15))) * numba.uint64(0x846CA68B)) & mask32
    x = (x ^ (x >> numba.uint64(16))) & mask32
    return float(x) / 4294967296.0


@numba.njit(parallel=True, cache=True)
def step_numba_parallel(
    state_in: np.ndarray,
    state_out: np.ndarray,
    timer_in: np.ndarray,
    timer_out: np.ndarray,
    toa: np.ndarray,
    elevation: np.ndarray,
    fuel: np.ndarray,
    mask_guatemala: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cell_size: float,
    next_time_hours: float,
    burn_duration_steps: int,
    step_idx: int,
    seed: int,
) -> None:
    """Ejecuta un paso temporal del automata celular en paralelo sobre filas (prange)."""
    rows, cols = state_in.shape
    c_slope = 2.5
    c_wind = 0.12
    inv_sqrt2 = 0.7071067811865475

    # Vecindad de Moore: (dr, dc, dist_factor)
    drs = (-1, -1, -1, 0, 0, 1, 1, 1)
    dcs = (-1, 0, 1, -1, 1, -1, 0, 1)
    dists = (1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356)
    inv_dists = (inv_sqrt2, 1.0, inv_sqrt2, 1.0, 1.0, inv_sqrt2, 1.0, inv_sqrt2)

    for r in numba.prange(1, rows - 1):
        for c in range(1, cols - 1):
            st = state_in[r, c]
            if st == STATE_BURNED:
                state_out[r, c] = STATE_BURNED
                timer_out[r, c] = timer_in[r, c]
                continue

            if st == STATE_BURNING:
                t_next = timer_in[r, c] + 1
                timer_out[r, c] = t_next
                if t_next >= burn_duration_steps:
                    state_out[r, c] = STATE_BURNED
                else:
                    state_out[r, c] = STATE_BURNING
                continue

            # Estado UNBURNED: verificar si es territorio valido y combustible
            if not mask_guatemala[r, c] or fuel[r, c] <= 0.08:
                state_out[r, c] = STATE_UNBURNED
                timer_out[r, c] = 0
                continue

            h_target = elevation[r, c]
            f_target = fuel[r, c]
            ignited = False

            for k in range(8):
                nr = r + drs[k]
                nc = c + dcs[k]
                if state_in[nr, nc] == STATE_BURNING and mask_guatemala[nr, nc]:
                    inv_d = inv_dists[k]
                    dh_norm = (h_target - elevation[nr, nc]) * inv_d / cell_size
                    if dh_norm > 1.0:
                        dh_norm = 1.0
                    elif dh_norm < -1.0:
                        dh_norm = -1.0

                    # Direccion DESDE vecino ardiendo (nr, nc) HACIA celda objetivo (r, c)
                    dir_x = -dcs[k] * inv_d
                    dir_y = drs[k] * inv_d  # Eje Y invertido (fila 0 = norte)
                    wind_dot = wind_u[nr, nc] * dir_x + wind_v[nr, nc] * dir_y

                    prob = 0.25 * f_target * math.exp(c_slope * dh_norm) * math.exp(c_wind * wind_dot)
                    if prob > 0.95:
                        prob = 0.95
                    elif prob < 0.0:
                        prob = 0.0

                    if _rand_u01(np.uint32(seed), np.int32(step_idx), np.int32(r), np.int32(c), np.int32(k)) < prob:
                        ignited = True
                        break

            if ignited:
                state_out[r, c] = STATE_BURNING
                timer_out[r, c] = 0
                toa[r, c] = next_time_hours
            else:
                state_out[r, c] = STATE_UNBURNED
                timer_out[r, c] = 0


def run_numba_simulation(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta N pasos completos con doble buffering en CPU paralela Numba."""
    state_a = state_init.astype(np.int32, copy=True)
    state_b = state_a.copy()
    timer_a = np.zeros_like(state_a)
    timer_b = np.zeros_like(state_a)
    toa = np.full(state_a.shape, np.nan, dtype=np.float32)
    toa[state_a == STATE_BURNING] = 0.0

    elev_f32 = np.ascontiguousarray(elevation, dtype=np.float32)
    fuel_f32 = np.ascontiguousarray(fuel, dtype=np.float32)
    mask_u8 = np.ascontiguousarray(mask_guatemala, dtype=np.uint8)
    wu_f32 = np.ascontiguousarray(wind_u, dtype=np.float32)
    wv_f32 = np.ascontiguousarray(wind_v, dtype=np.float32)

    for s in range(1, steps + 1):
        step_numba_parallel(
            state_a,
            state_b,
            timer_a,
            timer_b,
            toa,
            elev_f32,
            fuel_f32,
            mask_u8,
            wu_f32,
            wv_f32,
            float(cell_size),
            float(s * dt_hours),
            int(burn_duration_steps),
            int(s),
            int(seed),
        )
        state_a, state_b = state_b, state_a
        timer_a, timer_b = timer_b, timer_a

    return state_a, toa

