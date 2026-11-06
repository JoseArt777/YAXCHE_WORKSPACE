"""
YAXCHÉ - Motor de Simulación Celular 2D de Propagación de Frentes de Fuego
"""

import time
import numpy as np
from src.terrain import TerrainGrid
from src.wind import WindField


# Estados de la celda
STATE_UNBURNED = 0
STATE_BURNING = 1
STATE_BURNED = 2

# Desplazamientos para vecindad de Moore (8 vecinos) y sus distancias relativas
NEIGHBORS = [
    (-1, -1, np.sqrt(2.0)),  # Noroeste
    (-1,  0, 1.0),          # Norte
    (-1,  1, np.sqrt(2.0)),  # Noreste
    ( 0, -1, 1.0),          # Oeste
    ( 0,  1, 1.0),          # Este
    ( 1, -1, np.sqrt(2.0)),  # Suroeste
    ( 1,  0, 1.0),          # Sur
    ( 1,  1, np.sqrt(2.0)),  # Sureste
]


class FireSimulation:
    """
    Simulador de propagación física de incendios sobre mallas discretizadas.
    Implementación Baseline (Secuencial) para posterior perfilado y paralelización en HPC.
    """

    def __init__(self, terrain: TerrainGrid, wind: WindField):
        self.terrain = terrain
        self.wind = wind
        self.rows = terrain.rows
        self.cols = terrain.cols

        # Matriz de estados (0: no quemado, 1: ardiendo, 2: quemado)
        self.state = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Tiempo que una celda ha estado ardiendo
        self.burn_time = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Registro del paso temporal en que se quemó cada celda (isócronas)
        self.ignition_step = np.full((self.rows, self.cols), -1, dtype=np.int32)

        self.current_step = 0
        self.burn_duration = 3  # Pasos que tarda en consumirse el combustible

    def ignite(self, row: int, col: int):
        """Inicia un foco de ignición en la coordenada dada."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.state[row, col] = STATE_BURNING
            self.ignition_step[row, col] = self.current_step

    def step_sequential(self) -> int:
        """
        Ejecuta un paso de tiempo de la simulación de forma estrictamente secuencial.
        Este es el 'código víctima' base que será acelerado en Kabré con Numba y CUDA.
        Retorna el número de nuevas celdas encendidas.
        """
        new_ignitions = []
        c_slope = 2.0
        c_wind = 0.15

        # Recorrido de la malla (cuello de botella en CPU en mallas grandes)
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.state[r, c] == STATE_BURNING:
                    h_center = self.terrain.elevation[r, c]
                    w_u = self.wind.u[r, c]
                    w_v = self.wind.v[r, c]

                    for dr, dc, dist_factor in NEIGHBORS:
                        nr = r + dr
                        nc = c + dc

                        if self.state[nr, nc] == STATE_UNBURNED:
                            # 1. Factor de combustible
                            fuel = self.terrain.fuel[nr, nc]
                            if fuel <= 0.05:
                                continue  # Sin combustible suficiente

                            # 2. Factor de pendiente (cuesta arriba propaga más rápido)
                            h_neigh = self.terrain.elevation[nr, nc]
                            delta_h = (h_neigh - h_center) / (self.terrain.cell_size * dist_factor)
                            slope_factor = np.exp(c_slope * np.clip(delta_h, -1.0, 1.0))

                            # 3. Factor de viento (alineación con dirección del vecino)
                            # Vector dirección desde celda ardiendo hacia vecino (x=dc, y=-dr en coords cartesianas)
                            dir_x = dc / dist_factor
                            dir_y = -dr / dist_factor
                            wind_dot = (w_u * dir_x + w_v * dir_y)
                            wind_factor = np.exp(c_wind * wind_dot)

                            # Probabilidad neta de ignición
                            prob = 0.12 * fuel * slope_factor * wind_factor
                            prob = np.clip(prob, 0.0, 0.95)

                            if np.random.rand() < prob:
                                new_ignitions.append((nr, nc))

        # Actualizar estados de celdas recién encendidas
        for nr, nc in new_ignitions:
            if self.state[nr, nc] == STATE_UNBURNED:
                self.state[nr, nc] = STATE_BURNING
                self.ignition_step[nr, nc] = self.current_step + 1

        # Avanzar el tiempo de quemado de las celdas activas
        burning_mask = (self.state == STATE_BURNING)
        self.burn_time[burning_mask] += 1
        exhausted_mask = (self.burn_time >= self.burn_duration) & burning_mask
        self.state[exhausted_mask] = STATE_BURNED

        self.current_step += 1
        return len(new_ignitions)

    def run(self, steps: int = 20) -> list[int]:
        """Ejecuta una serie de pasos secuenciales y retorna historial de nuevas igniciones."""
        history = []
        for _ in range(steps):
            new_count = self.step_sequential()
            history.append(new_count)
        return history


def run_benchmark(sizes: list[int] = [200, 500, 1000], steps_per_size: int = 10):
    """
    Ejecuta un benchmark del baseline para documentar tiempos iniciales antes del taller.
    Evalúa múltiples focos de ignición sobre mallas de alta resolución.
    """
    print("=" * 75)
    print(" 🌳🔥 YAXCHÉ: Benchmark de Estrés en CPU Secuencial (Python Puro)")
    print("=" * 75)
    print(f"{'Malla (Filas x Cols)':<22} | {'Celdas':<12} | {'Memoria Est.':<14} | {'Tiempo (s)':<12} | {'s / paso':<10}")
    print("-" * 75)

    results = []
    for s in sizes:
        # Estimación de memoria de matrices principales (elevation, fuel, state, burn_time, wind)
        # ~6 matrices de float32 / int32 (4 bytes cada una)
        est_mb = (s * s * 4 * 6) / (1024 * 1024)

        t_gen0 = time.perf_counter()
        terrain = TerrainGrid.create_synthetic_guatemala(rows=s, cols=s)
        wind = WindField.uniform(shape=(s, s), speed_ms=12.0, direction_degrees=45.0)
        sim = FireSimulation(terrain, wind)

        # Iniciar 4 focos simultáneos (Petén, Altiplano, Oriente, Costa)
        sim.ignite(int(s * 0.8), int(s * 0.5))
        sim.ignite(int(s * 0.35), int(s * 0.45))
        sim.ignite(int(s * 0.5), int(s * 0.7))
        sim.ignite(int(s * 0.2), int(s * 0.2))

        t0 = time.perf_counter()
        sim.run(steps=steps_per_size)
        elapsed = time.perf_counter() - t0

        total_cells = s * s
        s_per_step = elapsed / steps_per_size
        print(f"{f'{s} x {s}':<22} | {total_cells:<12,d} | {f'{est_mb:.1f} MB':<14} | {elapsed:<12.4f} | {s_per_step:<10.4f}")
        results.append((s, total_cells, est_mb, elapsed, s_per_step))

    print("=" * 75)
    print("💡 Observación para Kabré:")
    print("   A 1 millón de celdas (1000x1000), el cuello de botella de los bucles 'for'")
    print("   en CPU monohilo se hace evidente. Con Numba y CUDA en Kabré, este tiempo")
    print("   se reducirá en órdenes de magnitud (10x a 50x de aceleración).")
    print("=" * 75)
    return results


if __name__ == "__main__":
    run_benchmark(sizes=[200, 500, 1000], steps_per_size=10)

