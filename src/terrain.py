"""
YAXCHE - Modulo de Modelado de Terreno y Georreferenciacion Nacional
Define la topografia, cobertura de combustible y mascara oficial de Guatemala.
Sistema de Coordenadas: WGS84 (EPSG:4326) / Metrica: 1 celda = 100m x 100m (1 ha).
"""

import numpy as np
from matplotlib.path import Path


class TerrainGrid:
    """
    Malla espacial georreferenciada del territorio de Guatemala.
    """

    # Bounding Box oficial de la Republica de Guatemala en grados decimales
    LON_MIN = -92.25
    LON_MAX = -88.22
    LAT_MIN = 13.73
    LAT_MAX = 17.82

    def __init__(
        self,
        elevation: np.ndarray,
        fuel: np.ndarray,
        mask_guatemala: np.ndarray,
        lon_coords: np.ndarray,
        lat_coords: np.ndarray,
        cell_size_meters: float = 100.0,
    ):
        """
        :param elevation: Matriz 2D de elevacion en metros s.n.m.
        :param fuel: Matriz 2D de densidad de combustible [0.0 - 1.0].
        :param mask_guatemala: Matriz booleana (True dentro de Guatemala, False fuera).
        :param lon_coords: Vector 1D de longitudes en grados decimales (°W).
        :param lat_coords: Vector 1D de latitudes en grados decimales (°N).
        :param cell_size_meters: Resolucion espacial de cada celda en metros.
        """
        self.elevation = elevation.astype(np.float32)
        self.fuel = fuel.astype(np.float32)
        self.mask_guatemala = mask_guatemala.astype(bool)
        self.lon_coords = lon_coords.astype(np.float32)
        self.lat_coords = lat_coords.astype(np.float32)
        self.cell_size = float(cell_size_meters)
        self.rows, self.cols = elevation.shape

        # Gradientes de pendiente en metros por metro
        self.slope_y, self.slope_x = np.gradient(self.elevation, self.cell_size)

    @classmethod
    def create_guatemala_domain(cls, rows: int = 400, cols: int = 400) -> "TerrainGrid":
        """
        Genera el dominio territorial georreferenciado con la silueta y orografia oficial de Guatemala:
        - Rectangulo de Peten y Tierras Bajas del Norte (0 a 300 m s.n.m.)
        - Cadena Volcanica de la Sierra Madre y Cuchumatanes (1500 a 4200 m s.n.m.)
        - Depresion del Rio Motagua y Lago de Izabal
        - Planicie costera hacia el Oceano Pacifico
        """
        lon = np.linspace(cls.LON_MIN, cls.LON_MAX, cols)
        lat = np.linspace(cls.LAT_MAX, cls.LAT_MIN, rows)  # De norte a sur (fila 0 = norte)
        LON, LAT = np.meshgrid(lon, lat)

        # -------------------------------------------------------------
        # Poligono vectorial simplificado de la frontera de Guatemala
        # (Vertices reales en WGS84)
        # -------------------------------------------------------------
        guatemala_polygon_coords = [
            (-90.99, 17.82),  # Vertice Noroeste de Peten (Frontera con Mexico)
            (-89.15, 17.82),  # Vertice Noreste de Peten (Frontera con Mexico y Belice)
            (-89.15, 15.89),  # Frontera Belice con Rio Sarstun
            (-88.60, 15.75),  # Bahia de Amatique / Puerto Barrios (Caribe)
            (-88.22, 15.35),  # Frontera con Honduras
            (-89.15, 14.45),  # Vertice Trifinio (Guatemala - Honduras - El Salvador)
            (-90.13, 13.74),  # Frontera El Salvador / Costa Pacifico (Ciudad Pedro de Alvarado)
            (-90.80, 13.92),  # Costa Pacifico (Puerto San Jose)
            (-91.50, 14.15),  # Costa Pacifico (Champerico)
            (-92.20, 14.51),  # Vertice Suroeste Pacifico / Mexico (Ocos)
            (-92.10, 15.15),  # Frontera Mexico / San Marcos (Volcan Tajumulco)
            (-91.80, 15.85),  # Frontera Mexico / Huehuetenango
            (-91.00, 16.50),  # Vertice Rio Usumacinta
            (-90.99, 17.20),  # Limite occidental de Peten
            (-90.99, 17.82),  # Cierre del poligono en el norte
        ]

        poly_path = Path(guatemala_polygon_coords)
        points = np.vstack((LON.flatten(), LAT.flatten())).T
        mask_guatemala = poly_path.contains_points(points).reshape((rows, cols))

        # -------------------------------------------------------------
        # Altimetria realista de Guatemala (Relieve Continental)
        # -------------------------------------------------------------
        # 1. Base topografica: gradiente sur a norte
        # Costa Pacifico (14.0° N) = 0m -> Planicie costera
        # Cadena volcanica paralela a la costa (14.4° - 14.9° N) = 2,500m - 4,200m
        # Altiplano y Cuchumatanes (15.0° - 15.8° N) = 2,000m - 3,800m
        # Valles del Motagua e Izabal (15.2° - 15.6° N, Lon -89.5°) = 100m - 400m
        # Tierras bajas de Peten (16.0° - 17.8° N) = 80m - 300m

        elevation = np.zeros((rows, cols), dtype=np.float32)

        # Planicie de Peten (Norte)
        peten_zone = (LAT >= 16.0)
        elevation[peten_zone] = 120.0 + 80.0 * np.sin(12 * LON[peten_zone]) * np.cos(10 * LAT[peten_zone])

        # Cadena Volcanica de la Sierra Madre (Sur-Centro)
        volcanic_axis = 14.65 - 0.22 * (LON - (-91.0))
        dist_volcanic = np.abs(LAT - volcanic_axis)
        ridge_volcanes = 3200.0 * np.exp(-(dist_volcanic**2) / (2 * 0.16**2))

        # Picos volcanicos prominentes
        tajumulco = 1000.0 * np.exp(-(((LON - (-91.90)) ** 2 + (LAT - 15.04) ** 2) / 0.015))
        fuego_acatenango = 800.0 * np.exp(-(((LON - (-90.88)) ** 2 + (LAT - 14.47) ** 2) / 0.012))
        agua = 600.0 * np.exp(-(((LON - (-90.74)) ** 2 + (LAT - 14.46) ** 2) / 0.010))

        # Sierra de los Cuchumatanes (Noroeste)
        cuchumatanes = 2600.0 * np.exp(-(((LON - (-91.50)) ** 2 + (LAT - 15.55) ** 2) / 0.08))

        # Depresion del Valle del Motagua (Valle seco central-este)
        motagua_axis = 15.0 + 0.35 * (LON - (-90.0))
        motagua_trough = 800.0 * np.exp(-((LAT - motagua_axis) ** 2) / 0.04) * (LON > -90.5)

        # Planicie costera del Pacifico (disminuye hacia el oceano en Lat < 14.2)
        elevation_montanias = ridge_volcanes + tajumulco + fuego_acatenango + agua + cuchumatanes - motagua_trough
        elevation_montanias = np.maximum(elevation_montanias, 50.0)

        elevation[~peten_zone] = elevation_montanias[~peten_zone]

        # Gradiente costero hacia el sur
        pacific_drop = np.clip((LAT - 13.9) / 0.45, 0.0, 1.0)
        elevation = elevation * pacific_drop

        # -------------------------------------------------------------
        # Cobertura de Combustible Forestal (Biomasa NDVI)
        # -------------------------------------------------------------
        fuel = np.zeros((rows, cols), dtype=np.float32)

        # Peten / Reserva de la Biosfera Maya: Bosque tropical denso (0.80 - 0.95)
        fuel[peten_zone] = 0.88 + 0.06 * np.cos(8 * LON[peten_zone])

        # Bosque de montania y coniferas en altiplano (0.60 - 0.75)
        altiplano = (~peten_zone) & (elevation > 1500)
        fuel[altiplano] = 0.70

        # Valles y costa: sabana y tierras agricolas (0.35 - 0.50)
        valles_costa = (~peten_zone) & (elevation <= 1500)
        fuel[valles_costa] = 0.42

        # Picos volcanicos rocosos sobre 3,600m: sin vegetacion
        fuel[elevation > 3600] = 0.05

        # Cero combustible y elevacion nula fuera de las fronteras soberanas de Guatemala
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
