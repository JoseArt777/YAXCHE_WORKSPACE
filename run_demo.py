#!/usr/bin/env python3
"""
YAXCHE - Script de Demostracion Estandar y Generacion de Reporte Cartografico
Ejecutalo con: python3 run_demo.py
"""

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

    rows, cols = 350, 350
    print(f"1. Generando dominio georreferenciado WGS84 ({rows} x {cols} = {rows*cols:,} celdas)...")
    print(f"   -> Escala metrica: 1 celda = 100m x 100m = 1 hectarea.")
    terrain = TerrainGrid.create_guatemala_domain(rows=rows, cols=cols)

    ha_guatemala = int(terrain.mask_guatemala.sum())
    print(f"   -> Territorio continental de Guatemala delimitado: {ha_guatemala:,} hectareas.")

    print("2. Configurando viento meteorologico NOAA GFS (14 km/h hacia el Noreste)...")
    wind = WindField.uniform(shape=(rows, cols), speed_ms=4.0, direction_degrees=180.0)

    print("3. Inicializando simulador...")
    sim = FireSimulation(terrain=terrain, wind=wind, step_duration_hours=1.0)

    # Iniciar foco en Peten (Reserva de la Biosfera Maya: Lon -90.50, Lat 17.20)
    # e ignicion secundaria en la Sierra Madre (Lon -91.10, Lat 14.55)
    print("4. Iniciando focos de ignicion reales:")
    print("   -> Foco 1: Reserva de la Biosfera Maya (Peten: 17.20°N, 90.50°W)")
    print("   -> Foco 2: Cadena Volcanica / Sierra Madre (14.55°N, 91.10°W)")
    sim.ignite_coord(lon=-89.0, lat=15.5)
    sim.ignite_coord(lon=-91.10, lat=14.55)

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
