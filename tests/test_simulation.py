"""
Pruebas de integracion del motor de simulacion YAXCHE
"""

import numpy as np
from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation, STATE_BURNING


def test_guatemala_domain():
    terrain = TerrainGrid.create_guatemala_domain(rows=100, cols=100)
    assert terrain.elevation.shape == (100, 100)
    assert terrain.fuel.shape == (100, 100)
    assert terrain.mask_guatemala.shape == (100, 100)
    assert np.any(terrain.mask_guatemala)  # Debe contener celdas en territorio guatemalteco


def test_coordinate_conversion():
    terrain = TerrainGrid.create_guatemala_domain(rows=100, cols=100)
    # Peten: Lon -90.5, Lat 17.2
    row, col = terrain.coord_to_indices(-90.5, 17.2)
    assert 0 <= row < 100
    assert 0 <= col < 100
    assert row < 50  # En el norte (mitad superior)


def test_fire_simulation_steps():
    terrain = TerrainGrid.create_guatemala_domain(rows=80, cols=80)
    wind = WindField.uniform(shape=(80, 80), speed_ms=8.0, direction_degrees=45.0)
    sim = FireSimulation(terrain, wind, step_duration_hours=1.0)

    sim.ignite_coord(lon=-90.5, lat=17.2)
    history = sim.run(steps=3)
    assert len(history) == 3
    assert sim.current_step == 3
    assert history[-1]["total_affected_ha"] > 0
