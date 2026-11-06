"""
YAXCHÉ - Módulo de Visualización Cartográfica y Térmica
"""

import matplotlib.pyplot as plt
import numpy as np
from src.simulation import FireSimulation, STATE_BURNING, STATE_BURNED


def plot_simulation_state(sim: FireSimulation, title: str = "YAXCHÉ - Simulación de Frentes de Fuego en Guatemala", save_path: str = None):
    """
    Renderiza el mapa de elevación del terreno junto con el frente térmico activo y zonas quemadas.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # 1. Mapa de elevación de fondo (Topografía)
    im_elev = ax.imshow(sim.terrain.elevation, cmap="terrain", origin="upper")
    cbar_elev = plt.colorbar(im_elev, ax=ax, fraction=0.046, pad=0.04)
    cbar_elev.set_label("Elevación (m.s.n.m.)", fontsize=10)

    # 2. Máscara de fuego
    fire_mask = np.zeros((*sim.state.shape, 4))  # RGBA

    # Celdas quemadas (ceniza / carbón en gris oscuro semi-transparente)
    burned = (sim.state == STATE_BURNED)
    fire_mask[burned] = [0.15, 0.15, 0.15, 0.75]

    # Celdas ardiendo (frente de fuego en rojo/naranja brillante)
    burning = (sim.state == STATE_BURNING)
    fire_mask[burning] = [1.0, 0.2, 0.0, 0.95]

    ax.imshow(fire_mask, origin="upper")

    # 3. Muestra de vectores de viento (quiver reducido para no saturar)
    step_skip = max(1, sim.rows // 25)
    y_idxs, x_idxs = np.mgrid[0:sim.rows:step_skip, 0:sim.cols:step_skip]
    u_sub = sim.wind.u[0:sim.rows:step_skip, 0:sim.cols:step_skip]
    v_sub = sim.wind.v[0:sim.rows:step_skip, 0:sim.cols:step_skip]

    ax.quiver(
        x_idxs, y_idxs, u_sub, -v_sub,  # -v porque el origen de imshow es 'upper'
        color="cyan", alpha=0.6, scale=150, width=0.002, label="Viento"
    )

    burned_count = int(np.sum(burned))
    burning_count = int(np.sum(burning))
    ax.set_title(
        f"{title}\nPaso temporal: {sim.current_step} | Frentes activos: {burning_count:,} celdas | Área quemada: {burned_count:,} celdas",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Coordenada X (Celdas)", fontsize=10)
    ax.set_ylabel("Coordenada Y (Celdas)", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[YAXCHÉ] Figura guardada con éxito en: {save_path}")

    return fig, ax

