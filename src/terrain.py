"""
YAXCHE - Modulo de Modelado de Terreno y Georreferenciacion Nacional
Utiliza las coordenadas oficiales de frontera de la Republica de Guatemala (Natural Earth / WGS84).
Metrica: 1 celda = 100m x 100m (1 ha).
"""

import json
import os
import numpy as np
from matplotlib.path import Path


# Coordenadas oficiales de frontera de la Republica de Guatemala (WGS84)
# Extraidas del estandar cartografico internacional Natural Earth (GTM)
OFFICIAL_GUATEMALA_BORDER = [
    (-90.095555, 13.735338),
    (-90.608624, 13.909771),
    (-91.232410, 13.927832),
    (-91.689747, 14.126218),
    (-92.227750, 14.538829),
    (-92.203230, 14.830103),
    (-92.087216, 15.064585),
    (-92.229249, 15.251447),
    (-91.747960, 16.066565),
    (-90.464473, 16.069562),
    (-90.438867, 16.410110),
    (-90.600847, 16.470778),
    (-90.711822, 16.687483),
    (-91.081670, 16.918477),
    (-91.453921, 17.252177),
    (-91.002269, 17.254658),
    (-91.001520, 17.817595),
    (-90.067934, 17.819326),
    (-89.143080, 17.808319),
    (-89.150806, 17.015577),
    (-89.229122, 15.886938),
    (-88.930613, 15.887273),
    (-88.604586, 15.706380),
    (-88.518364, 15.855389),
    (-88.225023, 15.727722),
    (-88.680680, 15.346247),
    (-89.154811, 15.066419),
    (-89.225220, 14.874286),
    (-89.145535, 14.678019),
    (-89.353326, 14.424133),
    (-89.587343, 14.362586),
    (-89.534219, 14.244816),
    (-89.721934, 14.134228),
    (-90.064678, 13.881970),
    (-90.095555, 13.735338),
]


class TerrainGrid:
    """
    Malla espacial georreferenciada con la frontera oficial de Guatemala.
    """

    # Bounding Box con margen geografico para visualizacion contextual
    LON_MIN = -92.50
    LON_MAX = -88.00
    LAT_MIN = 13.50
    LAT_MAX = 18.00

    def __init__(
        self,
        elevation: np.ndarray,
        fuel: np.ndarray,
        mask_guatemala: np.ndarray,
        lon_coords: np.ndarray,
        lat_coords: np.ndarray,
        cell_size_meters: float = 100.0,
    ):
        self.elevation = elevation.astype(np.float32)
        self.fuel = fuel.astype(np.float32)
        self.mask_guatemala = mask_guatemala.astype(bool)
        self.lon_coords = lon_coords.astype(np.float32)
        self.lat_coords = lat_coords.astype(np.float32)
        self.cell_size = float(cell_size_meters)
        self.rows, self.cols = elevation.shape
        self.border_coords = OFFICIAL_GUATEMALA_BORDER

        # Gradientes de pendiente
        self.slope_y, self.slope_x = np.gradient(self.elevation, self.cell_size)

    @classmethod
    def create_guatemala_domain(cls, rows: int = 400, cols: int = 400) -> "TerrainGrid":
        """
        Genera el dominio territorial oficial de Guatemala sobre la malla WGS84.
        """
        lon = np.linspace(cls.LON_MIN, cls.LON_MAX, cols)
        lat = np.linspace(cls.LAT_MAX, cls.LAT_MIN, rows)  # De norte a sur
        LON, LAT = np.meshgrid(lon, lat)

        # Mascara exacta del poligono oficial
        poly_path = Path(OFFICIAL_GUATEMALA_BORDER)
        points = np.vstack((LON.flatten(), LAT.flatten())).T
        mask_guatemala = poly_path.contains_points(points).reshape((rows, cols))

        # -------------------------------------------------------------
        # Relieve Topografico Georreferenciado
        # -------------------------------------------------------------
        elevation = np.zeros((rows, cols), dtype=np.float32)

        # 1. Tierras Bajas de Peten (Norte: Lat > 16.0)
        peten = (LAT >= 16.0)
        elevation[peten] = 120.0 + 60.0 * np.sin(10 * LON[peten]) * np.cos(8 * LAT[peten])

        # 2. Cadena Volcanica de la Sierra Madre (Sur: Lat 14.3 a 15.0)
        # Eje volcanico orientado NO-SE
        volc_center = 14.50 - 0.25 * (LON - (-91.0))
        dist_volc = np.abs(LAT - volc_center)
        volc_ridge = 3200.0 * np.exp(-(dist_volc**2) / (2 * 0.18**2))

        # Picos volcanicos destacados
        tajumulco = 1000.0 * np.exp(-(((LON - (-91.90)) ** 2 + (LAT - 15.04) ** 2) / 0.02))
        fuego_acatenango = 800.0 * np.exp(-(((LON - (-90.88)) ** 2 + (LAT - 14.47) ** 2) / 0.015))
        agua = 600.0 * np.exp(-(((LON - (-90.74)) ** 2 + (LAT - 14.46) ** 2) / 0.012))

        # 3. Sierra de los Cuchumatanes (Huehuetenango: Lat 15.5, Lon -91.5)
        cuchumatanes = 2800.0 * np.exp(-(((LON - (-91.50)) ** 2 + (LAT - 15.55) ** 2) / 0.07))

        # 4. Valle del Rio Motagua
        motagua_axis = 14.9 + 0.35 * (LON - (-90.0))
        motagua_trough = 700.0 * np.exp(-((LAT - motagua_axis) ** 2) / 0.05) * (LON > -90.5)

        elevation_montanias = volc_ridge + tajumulco + fuego_acatenango + agua + cuchumatanes - motagua_trough
        elevation_montanias = np.maximum(elevation_montanias, 80.0)

        elevation[~peten] = elevation_montanias[~peten]

        # Caida hacia el Pacifico (Lat < 14.1)
        pacific_slope = np.clip((LAT - 13.8) / 0.35, 0.0, 1.0)
        elevation = elevation * pacific_slope

        # -------------------------------------------------------------
        # Cobertura de Combustible Forestal (NDVI)
        # -------------------------------------------------------------
        fuel = np.zeros((rows, cols), dtype=np.float32)
        # Peten: Selva tropical de alta densidad (0.85 - 0.95)
        fuel[peten] = 0.90
        # Altiplano: Bosque mixto y coniferas (0.65 - 0.75)
        fuel[(~peten) & (elevation > 1400)] = 0.70
        # Costa y valles agricolas (0.40)
        fuel[(~peten) & (elevation <= 1400)] = 0.40
        # Picos volcanicos rocosos
        fuel[elevation > 3700] = 0.05

        # Restringir estrictamente al territorio soberano
        elevation[~mask_guatemala] = 0.0
        fuel[~mask_guatemala] = 0.0

        return cls(
            elevation=elevation,
            fuel=fuel,
            mask_guatemala=mask_guatemala,
            lon_coords=lon,
            lat_coords=lat,
            cell_size_meters=100.0,
        )

    def coord_to_indices(self, lon: float, lat: float) -> tuple[int, int]:
        """Convierte coordenadas geograficas (Lon, Lat) a indices de matriz (fila, columna)."""
        col = int(np.interp(lon, [self.LON_MIN, self.LON_MAX], [0, self.cols - 1]))
        row = int(np.interp(lat, [self.LAT_MIN, self.LAT_MAX], [self.rows - 1, 0]))
        return int(np.clip(row, 0, self.rows - 1)), int(np.clip(col, 0, self.cols - 1))

    def indices_to_coord(self, row: int, col: int) -> tuple[float, float]:
        """Convierte indices de matriz a coordenadas geograficas (Lon, Lat)."""
        lon = float(self.lon_coords[col])
        lat = float(self.lat_coords[row])
        return lon, lat
