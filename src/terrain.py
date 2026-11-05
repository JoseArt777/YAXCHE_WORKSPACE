"""
JURAKAN - Módulo de Modelado de Terreno y Combustible Forestal
"""

import numpy as np


class TerrainGrid:
    """
    Representa la malla espacial del terreno (elevación, pendientes y combustible).
    """

    def __init__(self, elevation: np.ndarray, fuel: np.ndarray, cell_size_meters: float = 100.0):
        """
        :param elevation: Matriz 2D de elevación en metros (DEM).
        :param fuel: Matriz 2D con coeficiente de combustible/biomasa en rango [0.0, 1.0].
        :param cell_size_meters: Resolución espacial de cada celda en metros.
        """
        if elevation.shape != fuel.shape:
            raise ValueError("Las matrices de elevación y combustible deben tener dimensiones idénticas.")

        self.elevation = elevation.astype(np.float32)
        self.fuel = np.clip(fuel, 0.0, 1.0).astype(np.float32)
        self.cell_size = float(cell_size_meters)
        self.rows, self.cols = elevation.shape

        # Precalcular gradientes de elevación (pendiente)
        self.slope_y, self.slope_x = np.gradient(self.elevation, self.cell_size)

    @classmethod
    def create_synthetic_guatemala(cls, rows: int = 500, cols: int = 500, cell_size_meters: float = 100.0) -> "TerrainGrid":
        """
        Genera una malla sintética inspirada en la topografía de Guatemala:
        - Norte (Petén): Llanuras bajas de selva densa (altos combustibles, elevación baja).
        - Centro/Occidente: Cadena montañosa volcánica (Sierra Madre / Cuchumatanes, alta elevación y pendientes).
        - Sur: Planicie costera del Pacífico.
        """
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        X, Y = np.meshgrid(x, y)

        # Elevación: cordillera central montañosa inclinada
        base_elevation = 200.0 + 800.0 * Y  # Gradiente sur a norte
        # Cordillera volcánica en el tercio sur-central (y entre 0.25 y 0.5)
        mountain_ridge = 2500.0 * np.exp(-((Y - 0.38) ** 2) / 0.02) * np.sin(np.pi * X) ** 2
        # Volcanes puntuales (picos cónicos)
        volcan_1 = 1500.0 * np.exp(-(((X - 0.45) ** 2 + (Y - 0.35) ** 2) / 0.003))
        volcan_2 = 1800.0 * np.exp(-(((X - 0.55) ** 2 + (Y - 0.33) ** 2) / 0.003))
        # Sierra de los Cuchumatanes (noroeste)
        cuchumatanes = 2000.0 * np.exp(-(((X - 0.28) ** 2 + (Y - 0.55) ** 2) / 0.015))

        # Ruido rugoso fractal simple
        noise = 50.0 * np.sin(10 * X) * np.cos(10 * Y)

        elevation = base_elevation + mountain_ridge + volcan_1 + volcan_2 + cuchumatanes + noise
        elevation = np.maximum(elevation, 0.0)

        # Combustible:
        # En el norte (Petén, Y > 0.65) selva muy densa (~0.85 - 0.95)
        # En montañas (centro) bosques templados (~0.6 - 0.75)
        # En valles agrícolas / urbes (~0.2 - 0.4)
        fuel = 0.5 + 0.35 * (Y > 0.65) - 0.2 * (elevation > 2800) + 0.1 * np.cos(20 * X) * np.sin(20 * Y)
        fuel = np.clip(fuel, 0.05, 0.95)

        return cls(elevation=elevation, fuel=fuel, cell_size_meters=cell_size_meters)
