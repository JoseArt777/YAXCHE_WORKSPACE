#!/usr/bin/env python3
"""
YAXCHE - Suite de Benchmarks Reproducibles y Ensamble Monte Carlo para Kabre (CeNAT)
Registra metricas estadisticas (mediana, IQR, first_call_s) en formato JSON Lines (.jsonl)
siguiendo las convenciones de course_hpc.measure.
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulation import STATE_BURNED, STATE_BURNING
from src.simulation_numba import run_numba_simulation
from src.terrain import TerrainGrid
from src.wind import WindField


def summarize_samples(samples: list[float]) -> dict:
    med = statistics.median(samples)
    if len(samples) >= 2:
        q = statistics.quantiles(samples, n=4, method="inclusive")
        q25, q75 = q[0], q[2]
    else:
        q25 = q75 = samples[0]
    return {
        "median_s": med,
        "best_s": min(samples),
        "iqr_low_s": q25,
        "iqr_high_s": q75,
        "samples_s": samples,
    }


def append_record(out_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    record["hostname"] = socket.gethostname()
    record["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def prepare_state_init(terrain: TerrainGrid) -> np.ndarray:
    state = np.zeros((terrain.rows, terrain.cols), dtype=np.int32)
    # Focos de ignicion en Peten (RBM) y Cadena Volcanica / Altiplano
    for lon, lat in [(-90.50, 17.20), (-91.10, 14.55)]:
        r, c = terrain.coord_to_indices(lon, lat)
        if terrain.mask_guatemala[r, c] and terrain.fuel[r, c] > 0.08:
            state[r, c] = STATE_BURNING
    return state


def run_cpu_benchmarks(terrain: TerrainGrid, wind: WindField, steps: int, repeat: int, out_jsonl: str):
    state_init = prepare_state_init(terrain)
    max_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", numba.config.NUMBA_NUM_THREADS))
    thread_counts = [t for t in [1, 2, 4, 8, 12, 16, 20] if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)

    print(f"[CPU] Malla {terrain.rows}x{terrain.cols}, pasos={steps}, hilos={thread_counts}")
    for nt in thread_counts:
        numba.set_num_threads(nt)
        t0 = time.perf_counter()
        st, _ = run_numba_simulation(
            state_init, terrain.elevation, terrain.fuel, terrain.mask_guatemala,
            wind.u, wind.v, terrain.cell_size, steps=steps, seed=42
        )
        first_call_s = time.perf_counter() - t0

        samples = []
        for _ in range(repeat):
            t1 = time.perf_counter()
            st, _ = run_numba_simulation(
                state_init, terrain.elevation, terrain.fuel, terrain.mask_guatemala,
                wind.u, wind.v, terrain.cell_size, steps=steps, seed=42
            )
            samples.append(time.perf_counter() - t1)

        stats = summarize_samples(samples)
        affected = int(np.sum((st == STATE_BURNING) | (st == STATE_BURNED)))
        rec = {
            "mode": "cpu_numba",
            "threads": nt,
            "grid": [terrain.rows, terrain.cols],
            "steps": steps,
            "first_call_s": first_call_s,
            "affected_cells": affected,
            **stats,
        }
        append_record(out_jsonl, rec)
        print(
            f"   -> threads={nt:2d} | median={stats['median_s']*1e3:8.2f} ms "
            f"(IQR {stats['iqr_low_s']*1e3:.2f}-{stats['iqr_high_s']*1e3:.2f} ms) | first={first_call_s*1e3:.1f} ms"
        )


def run_gpu_benchmarks(
    terrain: TerrainGrid,
    wind: WindField,
    steps: int,
    repeat: int,
    mc_runs: int,
    out_jsonl: str,
    out_fig: str,
):
    from src.simulation_gpu import HAS_CUPY, run_gpu_simulation
    if not HAS_CUPY:
        raise RuntimeError("CuPy no disponible; ejecute --mode gpu en un nodo GPU (nukwa-l40s / nukwa-v100).")

    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = props["name"].decode()
    print(f"[GPU] Dispositivo: {gpu_name} | Malla {terrain.rows}x{terrain.cols} | pasos={steps}")

    state_init = prepare_state_init(terrain)

    # 1. Comparacion coalescente (MAP_ROWS) vs strided y tuning de bloques
    configs = [
        (True, (16, 16)),
        (False, (16, 16)),
        (True, (32, 8)),
        (True, (32, 16)),
        (True, (8, 8)),
    ]

    for coalesced, bdim in configs:
        # Warmup (incluye compilacion NVRTC)
        t0 = time.perf_counter()
        res0 = run_gpu_simulation(
            state_init, terrain.elevation, terrain.fuel, terrain.mask_guatemala,
            wind.u, wind.v, terrain.cell_size, steps=steps, seed=42,
            block_dim=bdim, coalesced=coalesced
        )
        first_call_s = time.perf_counter() - t0

        kernel_samples, total_samples = [], []
        for _ in range(repeat):
            res = run_gpu_simulation(
                state_init, terrain.elevation, terrain.fuel, terrain.mask_guatemala,
                wind.u, wind.v, terrain.cell_size, steps=steps, seed=42,
                block_dim=bdim, coalesced=coalesced
            )
            kernel_samples.append(res["kernel_s"])
            total_samples.append(res["total_s"])

        k_stats = summarize_samples(kernel_samples)
        t_stats = summarize_samples(total_samples)
        affected = int(np.sum((res["state"] == STATE_BURNING) | (res["state"] == STATE_BURNED)))
        rec = {
            "mode": "gpu_cuda",
            "gpu_name": gpu_name,
            "coalesced": coalesced,
            "block_dim": list(bdim),
            "grid": [terrain.rows, terrain.cols],
            "steps": steps,
            "first_call_s": first_call_s,
            "kernel_median_s": k_stats["median_s"],
            "kernel_iqr": [k_stats["iqr_low_s"], k_stats["iqr_high_s"]],
            "total_median_s": t_stats["median_s"],
            "h2d_s": res["h2d_s"],
            "d2h_s": res["d2h_s"],
            "affected_cells": affected,
        }
        append_record(out_jsonl, rec)
        label = "coalesced" if coalesced else "strided  "
        print(
            f"   -> {label} block={bdim} | kernel={k_stats['median_s']*1e3:7.3f} ms "
            f"| end-to-end={t_stats['median_s']*1e3:7.3f} ms | affected={affected:,}"
        )

    # 2. Ensamble Estocastico Monte Carlo en GPU
    if mc_runs > 0:
        print(f"[GPU Monte Carlo] Ejecutando {mc_runs} replicas estocasticas sobre Guatemala...")
        burn_counts = np.zeros((terrain.rows, terrain.cols), dtype=np.float32)
        t_mc0 = time.perf_counter()
        rng = np.random.default_rng(2026)

        for r_idx in range(mc_runs):
            # Perturbacion meteorologica realista en viento (+- 15% velocidad, +- 15 deg)
            angle_rad = np.radians(rng.normal(0.0, 10.0))
            scale = rng.normal(1.0, 0.12)
            wu_pert = (wind.u * np.cos(angle_rad) - wind.v * np.sin(angle_rad)) * scale
            wv_pert = (wind.u * np.sin(angle_rad) + wind.v * np.cos(angle_rad)) * scale

            res = run_gpu_simulation(
                state_init, terrain.elevation, terrain.fuel, terrain.mask_guatemala,
                wu_pert.astype(np.float32), wv_pert.astype(np.float32),
                terrain.cell_size, steps=steps, seed=1000 + r_idx,
                block_dim=(32, 8), coalesced=True
            )
            burned_mask = (res["state"] == STATE_BURNING) | (res["state"] == STATE_BURNED)
            burn_counts += burned_mask.astype(np.float32)

        mc_total_s = time.perf_counter() - t_mc0
        prob_map = (burn_counts / float(mc_runs)) * 100.0
        append_record(out_jsonl, {
            "mode": "gpu_montecarlo",
            "gpu_name": gpu_name,
            "grid": [terrain.rows, terrain.cols],
            "steps": steps,
            "replicas": mc_runs,
            "total_s": mc_total_s,
            "s_per_replica": mc_total_s / mc_runs,
        })
        print(f"   -> {mc_runs} replicas completadas en {mc_total_s:.2f} s ({mc_total_s/mc_runs*1e3:.1f} ms/replica)")

        # Guardar mapa probabilistico de riesgo
        os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 9), dpi=160, facecolor="#F8FAFC")
        extent = [terrain.LON_MIN, terrain.LON_MAX, terrain.LAT_MIN, terrain.LAT_MAX]
        topo = np.copy(terrain.elevation)
        topo[~terrain.mask_guatemala] = np.nan
        ax.imshow(topo, extent=extent, origin="upper", cmap="terrain", alpha=0.65)
        prob_disp = np.where(prob_map > 1.0, prob_map, np.nan)
        im_p = ax.imshow(prob_disp, extent=extent, origin="upper", cmap="hot_r", vmin=0, vmax=100, alpha=0.85)
        border = terrain.border_coords
        ax.plot([p[0] for p in border], [p[1] for p in border], color="#0F172A", linewidth=1.6)
        cbar = plt.colorbar(im_p, ax=ax, fraction=0.036, pad=0.03)
        cbar.set_label("Probabilidad de Afectacion Monte Carlo (%)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_title(f"YAXCHE: Mapa Probabilistico de Riesgo ({mc_runs} Replicas en {gpu_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitud Oeste (WGS84)")
        ax.set_ylabel("Latitud Norte (WGS84)")
        plt.savefig(out_fig, bbox_inches="tight")
        plt.close()
        print(f"[OK] Mapa probabilistico guardado en: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Benchmarks HPC de YAXCHE para Cluster Kabre")
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--npz", type=str, default="data/processed/guatemala_real.npz")
    parser.add_argument("--size", type=int, default=0, help="Opcional: crear malla sintetica de NxN si no se usa --npz")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--mc-runs", type=int, default=200)
    parser.add_argument("--out", type=str, default="results/benchmarks_kabre.jsonl")
    parser.add_argument("--fig", type=str, default="results/riesgo_montecarlo.png")
    args = parser.parse_args()

    if args.size > 0:
        terrain = TerrainGrid.create_guatemala_domain(rows=args.size, cols=args.size)
        wind = WindField.uniform(shape=(args.size, args.size), speed_ms=4.0, direction_degrees=210.0)
    else:
        terrain = TerrainGrid.from_npz(args.npz)
        wind = WindField.from_npz(args.npz)

    if args.mode == "cpu":
        run_cpu_benchmarks(terrain, wind, args.steps, args.repeat, args.out)
    else:
        run_gpu_benchmarks(terrain, wind, args.steps, args.repeat, args.mc_runs, args.out, args.fig)


if __name__ == "__main__":
    main()

