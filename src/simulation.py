"""
YAXCHE - Motor de Simulacion Celular 2D de Propagacion de Frentes de Fuego
Implementa formulacion fisica de Rothermel, calculo de isocronas TOA (Time of Arrival)
y cuantificacion de hectareas consumidas.
"""

import time
import numpy as np
from src.terrain import TerrainGrid
from src.wind import WindField

# Estados discretos de celda
STATE_UNBURNED = 0
STATE_BURNING = 1
STATE_BURNED = 2

# Desplazamientos para vecindad de Moore (8 vecinos) con distancias euclidianas relativas
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
    Simulador estocastico de frentes de fuego con resolucion metrica en hectareas.
    """

    def __init__(self, terrain: TerrainGrid, wind: WindField, step_duration_hours: float = 1.0):
        """
        :param terrain: Objeto TerrainGrid georreferenciado con mascara territorial.
        :param wind: Objeto WindField con vectores de viento u, v.
        :param step_duration_hours: Duracion fisica de cada paso de simulacion en horas.
        """
        self.terrain = terrain
        self.wind = wind
        self.dt_hours = float(step_duration_hours)
        self.rows = terrain.rows
        self.cols = terrain.cols

        # Matriz de estados discretos
        self.state = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Tiempo de permanencia ardiendo (pasos)
        self.burn_duration_steps = 5
        self.burn_timer = np.zeros((self.rows, self.cols), dtype=np.int32)

        # Matriz de Isocronas: Tiempo de Llegada (Time of Arrival - TOA) en horas reales
        # Inicializado en NaN para celdas no afectadas
        self.time_of_arrival = np.full((self.rows, self.cols), np.nan, dtype=np.float32)

        # Matriz de Longitud de Llama estimada (metros)
        self.flame_length = np.zeros((self.rows, self.cols), dtype=np.float32)

        self.current_step = 0
        self.simulated_hours = 0.0

    def ignite_coord(self, lon: float, lat: float):
        """Inicia un foco de ignicion a partir de coordenadas geograficas WGS84."""
        row, col = self.terrain.coord_to_indices(lon, lat)
        self.ignite_cell(row, col)

    def ignite_cell(self, row: int, col: int):
        """Inicia un foco de ignicion en un indice de celda especifico."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.terrain.mask_guatemala[row, col] and self.terrain.fuel[row, col] > 0.05:
                self.state[row, col] = STATE_BURNING
                self.time_of_arrival[row, col] = self.simulated_hours
                self.flame_length[row, col] = 1.5  # Llama inicial estimada (m)

    def step_sequential(self) -> dict:
        """
        Ejecuta un paso de simulacion secuencial sobre la cuadricula de Guatemala.
        Retorna diccionario con metricas fisicas del paso.
        """
        new_ignitions = []
        c_slope = 2.5
        c_wind = 0.35
        scale_factor = float(np.clip(100.0 / self.terrain.cell_size, 0.15, 1.0))
        p_base = 0.18 * scale_factor

        # Recorrido de celdas ardiendo dentro de Guatemala
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.state[r, c] == STATE_BURNING and self.terrain.mask_guatemala[r, c]:
                    h_center = self.terrain.elevation[r, c]
                    w_u = self.wind.u[r, c]
                    w_v = self.wind.v[r, c]

                    for dr, dc, dist_factor in NEIGHBORS:
                        nr = r + dr
                        nc = c + dc

                        # Solo propaga dentro de las fronteras soberanas de Guatemala
                        if self.terrain.mask_guatemala[nr, nc] and self.state[nr, nc] == STATE_UNBURNED:
                            fuel = self.terrain.fuel[nr, nc]
                            if fuel <= 0.08:
                                continue  # Zona no combustible o roca

                            # Factor pendiente
                            h_neigh = self.terrain.elevation[nr, nc]
                            dh_normalized = (h_neigh - h_center) / (self.terrain.cell_size * dist_factor)
                            slope_factor = np.exp(c_slope * np.clip(dh_normalized, -1.0, 1.0))

                            # Factor viento
                            dir_x = dc / dist_factor
                            dir_y = -dr / dist_factor  # Invertir eje Y (fila 0 = norte)
                            wind_dot = (w_u * dir_x + w_v * dir_y)
                            wind_factor = np.exp(c_wind * wind_dot)

                            # Probabilidad estocastica de ignicion (Rothermel direccional)
                            prob = p_base * fuel * slope_factor * wind_factor
                            prob = np.clip(prob, 0.0, 0.95)

                            if np.random.rand() < prob:
                                new_ignitions.append((nr, nc))

        # Actualizar estado e isocronas
        next_time_hours = self.simulated_hours + self.dt_hours
        for nr, nc in new_ignitions:
            if self.state[nr, nc] == STATE_UNBURNED:
                self.state[nr, nc] = STATE_BURNING
                self.time_of_arrival[nr, nc] = next_time_hours
                # Longitud de llama en funcion del combustible y pendiente
                flame_m = 1.0 + 3.0 * self.terrain.fuel[nr, nc]
                self.flame_length[nr, nc] = flame_m

        # Transicion de celdas ardiendo a consumidas (ceniza)
        burning_mask = (self.state == STATE_BURNING)
        self.burn_timer[burning_mask] += 1
        exhausted_mask = burning_mask & (self.burn_timer >= self.burn_duration_steps)
        self.state[exhausted_mask] = STATE_BURNED

        self.current_step += 1
        self.simulated_hours = next_time_hours

        # Metricas estandar en Hectareas reales segun resolucion de la celda
        ha_per_cell = (self.terrain.cell_size / 100.0) ** 2
        burned_ha = int(round(int(np.sum(self.state == STATE_BURNED)) * ha_per_cell))
        active_ha = int(round(int(np.sum(self.state == STATE_BURNING)) * ha_per_cell))
        total_affected_ha = burned_ha + active_ha

        return {
            "step": self.current_step,
            "simulated_hours": self.simulated_hours,
            "new_ignitions_ha": len(new_ignitions),
            "active_front_ha": active_ha,
            "burned_consumed_ha": burned_ha,
            "total_affected_ha": total_affected_ha,
            "total_affected_km2": total_affected_ha / 100.0,
        }

    def run(self, steps: int = 24) -> list[dict]:
        """Ejecuta una serie de pasos secuenciales y retorna historial cuantitativo."""
        history = []
        for _ in range(steps):
            stats = self.step_sequential()
            history.append(stats)
        return history


def run_benchmark(sizes: list[int] = [200, 400, 800], steps: int = 10):
    """
    Benchmark estandar de rendimiento sobre el dominio georreferenciado de Guatemala.
    """
    print("=" * 80)
    print(" YAXCHÉ: Benchmark de Rendimiento en CPU Secuencial (Mascara Guatemala)")
    print("=" * 80)
    print(f"{'Malla (Filas x Cols)':<22} | {'Celdas':<12} | {'Hectareas Dom.':<16} | {'Tiempo (s)':<12} | {'s / paso':<10}")
    print("-" * 80)

    results = []
    for s in sizes:
        terrain = TerrainGrid.create_guatemala_domain(rows=s, cols=s)
        wind = WindField.uniform(shape=(s, s), speed_ms=12.0, direction_degrees=45.0)
        sim = FireSimulation(terrain, wind, step_duration_hours=1.0)

        # Iniciar 2 focos reales: Peten y Altiplano
        sim.ignite_coord(lon=-90.50, lat=17.20)  # Peten / Tikal
        sim.ignite_coord(lon=-91.10, lat=14.55)  # Cadena Volcanica

        t0 = time.perf_counter()
        sim.run(steps=steps)
        elapsed = time.perf_counter() - t0

        total_cells = s * s
        ha_domain = int(np.sum(terrain.mask_guatemala))
        s_per_step = elapsed / steps
        print(f"{f'{s} x {s}':<22} | {total_cells:<12,d} | {f'{ha_domain:,} ha':<16} | {elapsed:<12.4f} | {s_per_step:<10.4f}")
        results.append((s, total_cells, ha_domain, elapsed, s_per_step))

    print("=" * 80)
    print("Nota para Kabre:")
    print("  La eliminacion de celdas no pertenecientes a Guatemala reduce el calculo util,")
    print("  pero el acceso a memoria irregular en fronteras introduce desviaciones de hilos,")
    print("  el escenario optimo para demostrar coalescencia de memoria en kernels CUDA.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_benchmark(sizes=[200, 300, 400], steps=5)
