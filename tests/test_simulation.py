"""
Pruebas de integración del motor de simulación YAXCHÉ
"""

import numpy as np
from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation, STATE_BURNING, STATE_UNBURNED


def test_terrain_creation():
    terrain = TerrainGrid.create_synthetic_guatemala(rows=50, cols=50)
    assert terrain.elevation.shape == (50, 50)
    assert terrain.fuel.shape == (50, 50)
    assert np.all(terrain.fuel >= 0.0) and np.all(terrain.fuel <= 1.0)


def test_wind_field():
    wind = WindField.uniform(shape=(50, 50), speed_ms=12.0, direction_degrees=90.0)
    assert wind.speed.shape == (50, 50)
    assert np.isclose(wind.speed[0, 0], 12.0)


def test_simulation_step():
    terrain = TerrainGrid.create_synthetic_guatemala(rows=50, cols=50)
    wind = WindField.uniform(shape=(50, 50), speed_ms=10.0, direction_degrees=45.0)
    sim = FireSimulation(terrain, wind)

    sim.ignite(25, 25)
    assert sim.state[25, 25] == STATE_BURNING

    # Ejecutar 3 pasos
    history = sim.run(steps=3)
    assert len(history) == 3
    assert sim.current_step == 3

