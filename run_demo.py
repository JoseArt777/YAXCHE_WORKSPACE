#!/usr/bin/env python3
"""
YAXCHÉ - Script de Demostración Rápida y Generación de Imagen
Ejecútalo con: python3 run_demo.py o ./run_demo.py
"""

import time
import os
from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation
from src.visualization import plot_simulation_state


def main():
    print("=" * 60)
    print("🌳🔥 YAXCHÉ: Iniciando Simulación de Demostración")
    print("=" * 60)

    filas, cols = 300, 300
    print(f"1. Generando orografía sintética de Guatemala ({filas} x {cols} = {filas*cols:,} celdas)...")
    terreno = TerrainGrid.create_synthetic_guatemala(rows=filas, cols=cols, cell_size_meters=100.0)

    print("2. Configurando viento alisio de 12 m/s soplando hacia el Noreste (45°)...")
    viento = WindField.uniform(shape=(filas, cols), speed_ms=12.0, direction_degrees=45.0)

    print("3. Inicializando simulador e iniciando dos focos de ignición (Petén y Volcanes)...")
    sim = FireSimulation(terrain=terreno, wind=viento)
    sim.ignite(row=int(filas * 0.8), col=int(cols * 0.5))   # Foco en Petén
    sim.ignite(row=int(filas * 0.35), col=int(cols * 0.45))  # Foco en Cordillera

    pasos = 15
    print(f"4. Ejecutando {pasos} pasos temporales de propagación...")
    t0 = time.perf_counter()
    historial = sim.run(steps=pasos)
    tiempo_total = time.perf_counter() - t0

    print(f"   -> Completado en {tiempo_total:.4f} s ({tiempo_total/pasos:.4f} s/paso)")
    print(f"   -> Historial de nuevas celdas encendidas por paso: {historial}")

    # Guardar mapa en imagen PNG
    output_img = "fuego_guatemala.png"
    print(f"5. Generando visualización y guardando en '{output_img}'...")
    plot_simulation_state(sim, title="YAXCHÉ - Simulación de Frentes de Fuego en Guatemala", save_path=output_img)

    print("=" * 60)
    print(f"✅ ¡Demostración finalizada! Revisa la imagen generada: {os.path.abspath(output_img)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
