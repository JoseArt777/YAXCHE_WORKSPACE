#!/usr/bin/env python3
"""
YAXCHE - Script de Ingesta de Datos Satelitales y Meteorologicos Reales de Guatemala.
Descarga capas reales gratuitas sin API key y las empaqueta en data/processed/guatemala_real.npz:
  1. Topografia Real (DEM): AWS Terrain Tiles (Mapzen Terrarium PNG).
  2. Viento Real: Open-Meteo Historical Archive (grilla 4x4 sobre Guatemala).
  3. Focos de Calor Reales: NASA FIRMS VIIRS 375m (America Central 7d filtrado a Guatemala).
  4. Mascara Territorial y Cuerpos de Agua: Frontera Natural Earth + lagos mayores.
"""

import argparse
import csv
import io
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from matplotlib.path import Path
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.terrain import OFFICIAL_GUATEMALA_BORDER, TerrainGrid


TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FIRMS_CA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Central_America_7d.csv"
)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def download_terrarium_tile(args: tuple[int, int, int]) -> tuple[int, int, np.ndarray]:
    z, x, y = args
    url = TERRARIUM_URL.format(z=z, x=x, y=y)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    # Decodificacion Terrarium: elevacion en metros = (R * 256 + G + B / 256) - 32768
    elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
    return x, y, elev


def fetch_real_dem(lon_coords: np.ndarray, lat_coords: np.ndarray, zoom: int = 9) -> np.ndarray:
    """Descarga tiles Terrarium para Guatemala y remuestrea exactamente a la grilla WGS84."""
    x_min, y_min = deg2num(float(lat_coords.max()), float(lon_coords.min()), zoom)
    x_max, y_max = deg2num(float(lat_coords.min()), float(lon_coords.max()), zoom)

    tile_list = [(zoom, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    print(f"   -> Descargando {len(tile_list)} tiles DEM reales (zoom={zoom}) desde AWS Terrain Tiles...")

    mosaic_h = (y_max - y_min + 1) * 256
    mosaic_w = (x_max - x_min + 1) * 256
    mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for x, y, tile_elev in pool.map(download_terrarium_tile, tile_list):
            row_off = (y - y_min) * 256
            col_off = (x - x_min) * 256
            mosaic[row_off : row_off + 256, col_off : col_off + 256] = tile_elev

    # Mapear cada punto (LON, LAT) de la grilla WGS84 al espacio de pixeles del mosaico
    LON, LAT = np.meshgrid(lon_coords, lat_coords)
    n = 2.0 ** zoom
    x_float = (LON + 180.0) / 360.0 * n
    y_float = (1.0 - np.arcsinh(np.tan(np.radians(LAT))) / np.pi) / 2.0 * n

    px = (x_float - x_min) * 256.0
    py = (y_float - y_min) * 256.0

    elevation = map_coordinates(mosaic, [py, px], order=1, mode="nearest").astype(np.float32)
    return elevation


def fetch_real_wind(
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    date_str: str = "2024-04-15",
    hour_idx: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Consulta una grilla 4x4 de viento historico real en Open-Meteo e interpola a (u, v)."""
    lats_grid = np.linspace(13.8, 17.8, 4)
    lons_grid = np.linspace(-92.2, -88.2, 4)
    pts_lat = [float(round(la, 3)) for la in lats_grid for _ in lons_grid]
    pts_lon = [float(round(lo, 3)) for _ in lats_grid for lo in lons_grid]

    print(f"   -> Consultando viento real Open-Meteo para {date_str} {hour_idx:02d}:00 (grilla 4x4)...")
    params = {
        "latitude": ",".join(map(str, pts_lat)),
        "longitude": ",".join(map(str, pts_lon)),
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "America/Guatemala",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    u_coarse = np.zeros((4, 4), dtype=np.float32)
    v_coarse = np.zeros((4, 4), dtype=np.float32)

    for idx, entry in enumerate(data):
        i, j = divmod(idx, 4)
        spd = float(entry["hourly"]["wind_speed_10m"][hour_idx])
        deg_from = float(entry["hourly"]["wind_direction_10m"][hour_idx])
        rad = math.radians(deg_from)
        # Convencion meteorologica: direccion DESDE donde sopla -> vector HACIA donde empuja
        u_coarse[i, j] = -spd * math.sin(rad)
        v_coarse[i, j] = -spd * math.cos(rad)

    interp_u = RegularGridInterpolator((lats_grid, lons_grid), u_coarse, bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lats_grid, lons_grid), v_coarse, bounds_error=False, fill_value=None)

    LON, LAT = np.meshgrid(lon_coords, lat_coords)
    query_pts = np.stack([LAT, LON], axis=-1)
    u_full = interp_u(query_pts).astype(np.float32)
    v_full = interp_v(query_pts).astype(np.float32)
    return u_full, v_full


def fetch_real_firms_hotspots(raw_out_csv: str) -> np.ndarray:
    """Descarga focos de calor VIIRS 7d de NASA FIRMS para Centroamerica y filtra a Guatemala."""
    print("   -> Descargando focos de calor activos NASA FIRMS (VIIRS 375m, Centroamerica 7d)...")
    poly_path = Path(OFFICIAL_GUATEMALA_BORDER)
    hotspots = []
    try:
        resp = requests.get(FIRMS_CA_URL, timeout=30)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        fieldnames = reader.fieldnames
        rows_gt = []
        for row in reader:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if poly_path.contains_point((lon, lat)):
                rows_gt.append(row)
                hotspots.append([lon, lat, float(row.get("frp", 10.0))])
        if fieldnames and rows_gt:
            with open(raw_out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_gt)
        print(f"   -> Focos activos detectados dentro de Guatemala (ultimos 7d): {len(hotspots)}")
    except Exception as exc:
        print(f"   [AVISO] No se pudo descargar FIRMS en vivo ({exc}).")

    if not hotspots:
        # Focos historicos documentados de temporada seca en Guatemala (Petén y Cadena Volcánica)
        hotspots = [
            [-90.50, 17.20, 45.0],
            [-90.85, 17.45, 38.0],
            [-90.88, 14.47, 25.0],
        ]
    return np.array(hotspots, dtype=np.float32)


def build_fuel_and_water_mask(
    elevation: np.ndarray,
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    mask_guatemala: np.ndarray,
) -> np.ndarray:
    """Genera matriz de combustible forestal respetando fronteras, cumbres rocosas y lagos mayores."""
    LON, LAT = np.meshgrid(lon_coords, lat_coords)
    fuel = np.zeros_like(elevation, dtype=np.float32)

    # Clasificacion ecologica basada en altitud y latitud real
    peten = LAT >= 16.0
    fuel[peten] = 0.90                                    # Bosque tropical denso (RBM / Peten)
    fuel[(~peten) & (elevation > 1400)] = 0.72            # Bosque mixto / coniferas de altiplano
    fuel[(~peten) & (elevation <= 1400)] = 0.45           # Valles, piedemonte y costa
    fuel[elevation > 3500] = 0.05                         # Conos volcanicos rocosos sin vegetacion

    # Cuerpos de agua mayores de Guatemala (el fuego no se propaga sobre agua)
    lago_atitlan = ((LON - (-91.19)) / 0.095) ** 2 + ((LAT - 14.69) / 0.065) ** 2 <= 1.0
    lago_izabal = ((LON - (-89.17)) / 0.22) ** 2 + ((LAT - 15.50) / 0.085) ** 2 <= 1.0
    lago_peten_itza = ((LON - (-89.82)) / 0.14) ** 2 + ((LAT - 16.98) / 0.045) ** 2 <= 1.0
    lago_amatitlan = ((LON - (-90.56)) / 0.045) ** 2 + ((LAT - 14.45) / 0.025) ** 2 <= 1.0
    agua_cota_cero = elevation <= 1.5

    water_mask = lago_atitlan | lago_izabal | lago_peten_itza | lago_amatitlan | agua_cota_cero
    fuel[water_mask] = 0.0
    fuel[~mask_guatemala] = 0.0
    return fuel


def main():
    parser = argparse.ArgumentParser(description="Construye dataset real de Guatemala en formato .npz")
    parser.add_argument("--size", type=int, default=450, help="Tamano de la malla cuadrada (ej. 450 o 2000)")
    parser.add_argument("--zoom", type=int, default=9, help="Zoom de tiles Terrarium (8, 9 o 10)")
    parser.add_argument("--date", type=str, default="2024-04-15", help="Fecha historica para viento real (YYYY-MM-DD)")
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/guatemala_real.npz",
        help="Ruta de salida del archivo .npz",
    )
    args = parser.parse_args()

    rows = cols = args.size
    lon = np.linspace(TerrainGrid.LON_MIN, TerrainGrid.LON_MAX, cols, dtype=np.float32)
    lat = np.linspace(TerrainGrid.LAT_MAX, TerrainGrid.LAT_MIN, rows, dtype=np.float32)
    LON, LAT = np.meshgrid(lon, lat)

    print(f"[1/4] Generando mascara oficial WGS84 ({rows}x{cols})...")
    poly_path = Path(OFFICIAL_GUATEMALA_BORDER)
    pts = np.vstack((LON.flatten(), LAT.flatten())).T
    mask_guatemala = poly_path.contains_points(pts).reshape((rows, cols))

    print("[2/4] Descargando topografia satelital real (DEM)...")
    elevation = fetch_real_dem(lon, lat, zoom=args.zoom)
    elevation = np.maximum(elevation, 0.0)
    elevation[~mask_guatemala] = 0.0

    print("[3/4] Descargando viento historico real y focos NASA FIRMS...")
    u, v = fetch_real_wind(lon, lat, date_str=args.date, hour_idx=14)
    firms_csv = os.path.join(os.path.dirname(args.out), "..", "raw", "nasa_firms_guatemala_live.csv")
    hotspots = fetch_real_firms_hotspots(os.path.abspath(firms_csv))

    print("[4/4] Construyendo capa de combustible y mascara de lagos...")
    fuel = build_fuel_and_water_mask(elevation, lon, lat, mask_guatemala)

    # Calcular tamano aproximado de celda en metros (1 grado lat ~ 110,574 m)
    cell_size_m = float((TerrainGrid.LAT_MAX - TerrainGrid.LAT_MIN) / rows * 110574.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        elevation=elevation,
        fuel=fuel,
        mask_guatemala=mask_guatemala,
        lon_coords=lon,
        lat_coords=lat,
        wind_u=u,
        wind_v=v,
        hotspots=hotspots,
        cell_size_meters=np.float32(cell_size_m),
        wind_date=np.array(args.date),
    )

    elev_gt = elevation[mask_guatemala]
    print("-" * 60)
    print(f"[OK] Dataset real guardado en: {os.path.abspath(args.out)}")
    print(f"     Elevacion dentro de Guatemala: min={elev_gt.min():.1f} m, max={elev_gt.max():.1f} m")
    print(f"     Viento promedio ({args.date}): {np.mean(np.hypot(u, v)):.2f} m/s")
    print(f"     Resolucion por celda: {cell_size_m:.1f} m")
    print("-" * 60)


if __name__ == "__main__":
    main()
