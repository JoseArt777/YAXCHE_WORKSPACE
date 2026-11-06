"""
YAXCHÉ - Módulo de Dinámica de Viento y Factores Atmosféricos
"""

import numpy as np


class WindField:
    """
    Representa el campo vectorial de viento sobre el dominio de simulación.
    """

    def __init__(self, u: np.ndarray, v: np.ndarray):
        """
        :param u: Componente zonal del viento (hacia el este) en m/s.
        :param v: Componente meridional del viento (hacia el norte) en m/s.
        """
        if u.shape != v.shape:
            raise ValueError("Las matrices de componentes u y v deben tener dimensiones idénticas.")

        self.u = u.astype(np.float32)
        self.v = v.astype(np.float32)
        self.speed = np.sqrt(self.u ** 2 + self.v ** 2)
        # Ángulo en radianes respecto al eje X
        self.angle = np.arctan2(self.v, self.u)

    @classmethod
    def uniform(cls, shape: tuple[int, int], speed_ms: float = 8.0, direction_degrees: float = 45.0) -> "WindField":
        """
        Genera un campo de viento uniforme sobre la malla.
        :param direction_degrees: Dirección hacia donde sopla el viento (0° = Este, 90° = Norte).
        """
        rad = np.radians(direction_degrees)
        u_val = speed_ms * np.cos(rad)
        v_val = speed_ms * np.sin(rad)

        u = np.full(shape, u_val, dtype=np.float32)
        v = np.full(shape, v_val, dtype=np.float32)
        return cls(u=u, v=v)

