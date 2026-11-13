#!/usr/bin/env python3
"""
YAXCHE - Script de Demostracion Estandar y Generacion de Reporte Cartografico
Ejecutalo con: python3 run_demo.py
"""

import csv
import time
import os
from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation
from src.visualization import plot_scientific_fire_report


def main():
    print("=" * 70)
    print(" YAXCHÉ: Iniciando Simulacion Estandar de Incendios en Guatemala")
    print("=" * 70)

    real_npz = "data/processed/guatemala_real.npz"
    if os.path.exists(real_npz):
        print(f"1. Cargando dataset real de Guatemala desde '{real_npz}'...")
        terrain = TerrainGrid.from_npz(real_npz)
        wind = WindField.from_npz(real_npz)
        print(f"   -> Malla real WGS84 ({terrain.rows} x {terrain.cols} = {terrain.rows * terrain.cols:,} celdas).")
        print(f"   -> Resolucion aproximada por celda: {terrain.cell_size:.1f} m.")
    else:
        rows, cols = 350, 350
        print(f"1. Generando dominio georreferenciado WGS84 ({rows} x {cols} = {rows*cols:,} celdas)...")
        print("   -> Escala metrica: 1 celda = 100m x 100m = 1 hectarea.")
        terrain = TerrainGrid.create_guatemala_domain(rows=rows, cols=cols)
        wind = WindField.uniform(shape=(rows, cols), speed_ms=4.0, direction_degrees=180.0)

    ha_guatemala = int(terrain.mask_guatemala.sum())
    print(f"   -> Territorio continental de Guatemala delimitado: {ha_guatemala:,} celdas.")
    print(f"2. Campo de viento activo (velocidad media: {float(wind.speed.mean()) * 3.6:.1f} km/h)...")

    print("3. Inicializando simulador...")
    sim = FireSimulation(terrain=terrain, wind=wind, step_duration_hours=1.0)

    firms_csv = "data/raw/nasa_firms_viirs_guatemala_2024.csv"
    hotspots = []
    if os.path.exists(firms_csv):
        with open(firms_csv, newline="", encoding="utf-8") as f:
            candidates = [
                (float(r["longitude"]), float(r["latitude"]), float(r.get("frp", 0.0)), r.get("region_nombre", "Guatemala"))
                for r in csv.DictReader(f)
                if r.get("acq_date") == "2024-04-18" and r.get("confidence") == "high" and float(r.get("frp", 0.0)) >= 55.0
            ]
        candidates.sort(key=lambda item: item[2], reverse=True)
        for lon, lat, frp, region in candidates:
            if all(((lon - s[0]) ** 2 + (lat - s[1]) ** 2) ** 0.5 >= 0.18 for s in hotspots):
                hotspots.append((lon, lat, frp, region))
                if len(hotspots) >= 24:
                    break
    if not hotspots:
        hotspots = [(-90.50, 17.20, 80.0, "Peten RBM"), (-91.10, 14.55, 75.0, "Cadena Volcanica")]

    print(f"4. Iniciando {len(hotspots)} focos reales NASA FIRMS VIIRS (2024-04-18):")
    ignited_count = 0
    for lon, lat, frp, region in hotspots:
        sim.ignite_coord(lon=lon, lat=lat)
        ignited_count += 1
        if ignited_count <= 5:
            print(f"   -> {region}: ({lat:.2f} N, {lon:.2f} W) FRP={frp:.1f} MW")
    if ignited_count > 5:
        print(f"   -> ... y {ignited_count - 5} focos adicionales activos.")

    pasos = 48
    print(f"5. Ejecutando simulacion secuencial de {pasos} horas de propagacion...")
    t0 = time.perf_counter()
    history = sim.run(steps=pasos)
    tiempo_total = time.perf_counter() - t0

    stats_final = history[-1]
    print(f"   -> Tiempo de calculo CPU: {tiempo_total:.4f} s ({tiempo_total/pasos:.4f} s/hora simulada)")
    print(f"   -> Hectareas totales consumidas: {stats_final['total_affected_ha']:,} ha ({stats_final['total_affected_km2']:.2f} km²)")
    print(f"   -> Frentes activos en combustion: {stats_final['active_front_ha']:,} ha")

    # Guardar reporte cartografico estandar
    output_img = "fuego_guatemala.png"
    print(f"6. Generando reporte cartografico FARSITE/EFFIS en '{output_img}'...")
    plot_scientific_fire_report(sim, history=history, save_path=output_img)

    print("=" * 70)
    print(f"[OK] Reporte generado exitosamente en: {os.path.abspath(output_img)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
