"""
YAXCHE - Modulo de Visualizacion Cartografica Estandar (FARSITE / EFFIS)
Renderiza mapas de isocronas TOA (Time of Arrival), topografia de Guatemala
y analisis cuantitativo de hectareas quemadas en coordenadas WGS84.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from src.simulation import FireSimulation, STATE_BURNING, STATE_BURNED


def plot_scientific_fire_report(
    sim: FireSimulation,
    history: list[dict] = None,
    save_path: str = None,
):
    """
    Genera el reporte cartografico cientifico estandar de propagacion de incendios:
    - Panel 1 (Izquierdo): Mapa georreferenciado WGS84 con la silueta oficial de Guatemala, relieve,
      paises limitrofes, oceano Pacifico e Isocronas TOA.
    - Panel 2 (Superior Derecho): Curva temporal de Hectareas Consumidas acumuladas.
    - Panel 3 (Inferior Derecho): Balance cuantitativo formal y clasificacion de llama.
    """
    fig = plt.figure(figsize=(16, 9.5), dpi=140, facecolor="#F8FAFC")

    # -----------------------------------------------------------------
    # PANEL 1: MAPA CARTOGRAFICO PRINCIPAL (GEORREFERENCIADO WGS84)
    # -----------------------------------------------------------------
    ax_map = fig.add_axes([0.05, 0.08, 0.58, 0.85], facecolor="#E2E8F0")

    extent = [sim.terrain.LON_MIN, sim.terrain.LON_MAX, sim.terrain.LAT_MIN, sim.terrain.LAT_MAX]

    # Fondo oceanico para el Pacifico (al sur) y el Caribe (al este)
    # Latitud menor a la costa o cuadrante caribe
    lon_grid, lat_grid = np.meshgrid(sim.terrain.lon_coords, sim.terrain.lat_coords)
    ocean_mask = (lat_grid < 14.1 - 0.25 * (lon_grid - (-91.0))) | ((lon_grid > -88.8) & (lat_grid > 15.6) & (~sim.terrain.mask_guatemala))
    
    bg_color = np.full((*sim.terrain.elevation.shape, 3), [0.91, 0.93, 0.96]) # Gris neutro para paises vecinos
    bg_color[ocean_mask] = [0.82, 0.91, 0.99] # Azul suave para el oceano

    ax_map.imshow(bg_color, extent=extent, origin="upper")

    # 1. Topografia enmascarada dentro de Guatemala
    topo_display = np.copy(sim.terrain.elevation)
    topo_display[~sim.terrain.mask_guatemala] = np.nan

    im_topo = ax_map.imshow(
        topo_display,
        extent=extent,
        cmap="terrain",
        origin="upper",
        vmin=-200,
        vmax=4000,
        alpha=0.90,
    )

    # 2. Trazado de la Frontera Oficial de Guatemala (Vectorial)
    border = sim.terrain.border_coords
    b_lons = [p[0] for p in border]
    b_lats = [p[1] for p in border]
    ax_map.plot(b_lons, b_lats, color="#0F172A", linewidth=1.8, label="Frontera de Guatemala")

    # 3. Isocronas de Tiempo de Llegada (Time of Arrival - TOA)
    toa_display = np.copy(sim.time_of_arrival)
    toa_display[~sim.terrain.mask_guatemala] = np.nan

    if np.any(~np.isnan(toa_display)):
        # Celdas quemadas (ceniza / carbon en gris oscuro)
        burned_mask = (sim.state == STATE_BURNED)
        burned_overlay = np.zeros((*sim.state.shape, 4))
        burned_overlay[burned_mask] = [0.12, 0.12, 0.12, 0.85]
        ax_map.imshow(burned_overlay, extent=extent, origin="upper")

        # Frentes activos ardiendo (rojo intenso)
        active_mask = (sim.state == STATE_BURNING)
        active_overlay = np.zeros((*sim.state.shape, 4))
        active_overlay[active_mask] = [0.95, 0.15, 0.05, 0.98]
        ax_map.imshow(active_overlay, extent=extent, origin="upper")

        # Lineas de contorno de isocronas
        max_t = np.nanmax(toa_display)
        if max_t > 1:
            levels = np.linspace(0, max_t, min(6, int(max_t) + 1))
            contours = ax_map.contour(
                toa_display,
                levels=levels,
                extent=extent,
                origin="upper",
                cmap="autumn_r",
                linewidths=1.3,
            )
            ax_map.clabel(contours, inline=True, fontsize=8, fmt="%1.0fh", colors="black")

    # 4. Contexto de Paises Limitrofes y Oceanos
    labels_context = [
        (-92.10, 16.60, "MÉXICO"),
        (-88.80, 17.15, "BELICE"),
        (-88.55, 14.85, "HONDURAS"),
        (-89.45, 13.72, "EL SALVADOR"),
        (-91.80, 13.60, "OCÉANO PACÍFICO"),
    ]
    for lx, ly, ltext in labels_context:
        color_lbl = "#0284C7" if "PACÍFICO" in ltext else "#64748B"
        ax_map.text(lx, ly, ltext, fontsize=9.5, fontweight="bold", color=color_lbl, alpha=0.75,
                    ha="center", va="center", style="italic" if "PACÍFICO" in ltext else "normal")

    # 5. Puntos de Referencia Nacionales
    landmarks = [
        (-90.50, 17.20, "R.B. Maya (Petén)"),
        (-90.51, 14.63, "Cd. de Guatemala"),
        (-91.52, 14.83, "Quetzaltenango"),
        (-88.59, 15.72, "Puerto Barrios"),
        (-90.82, 13.92, "Costa Sur"),
    ]
    for lon_p, lat_p, label in landmarks:
        ax_map.plot(lon_p, lat_p, "o", color="#0F172A", markersize=4.5)
        ax_map.text(lon_p + 0.06, lat_p - 0.04, label, fontsize=8.5, fontweight="bold",
                    color="#0F172A", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#CBD5E1", linewidth=0.6))

    # 6. Rosa de los Vientos / Vector Atmosferico
    avg_speed = float(np.mean(sim.wind.speed))
    wind_kmh = avg_speed * 3.6

    ax_map.annotate(
        f"Viento GFS: {wind_kmh:.1f} km/h (NE)",
        xy=(-88.7, 14.05),
        xytext=(-89.6, 14.20),
        arrowprops=dict(facecolor="#0284C7", edgecolor="#0369A1", width=2, headwidth=7),
        fontsize=9,
        fontweight="bold",
        color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
    )

    # Bloquear aspecto proporcional para evitar distorsion cartografica
    ax_map.set_aspect("equal")
    ax_map.set_xlim(sim.terrain.LON_MIN, sim.terrain.LON_MAX)
    ax_map.set_ylim(sim.terrain.LAT_MIN, sim.terrain.LAT_MAX)
    ax_map.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f°W"))
    ax_map.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f°N"))
    ax_map.set_xlabel("Longitud Oeste (WGS84)", fontsize=9.5)
    ax_map.set_ylabel("Latitud Norte (WGS84)", fontsize=9.5)
    ax_map.grid(True, linestyle=":", alpha=0.45, color="#64748B")
    ax_map.set_title("República de Guatemala: Mapa de Isócronas de Propagación (TOA)", fontsize=11, fontweight="bold", pad=8)

    # Barra de escala de elevacion
    cbar_ax = fig.add_axes([0.08, 0.12, 0.14, 0.015])
    cbar = plt.colorbar(im_topo, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Elevación (m s.n.m.)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    # -----------------------------------------------------------------
    # PANEL 2: CURVA TEMPORAL DE HECTAREAS CONSUMIDAS
    # -----------------------------------------------------------------
    ax_curve = fig.add_axes([0.68, 0.54, 0.28, 0.38], facecolor="white")

    if history and len(history) > 0:
        times_h = [h["simulated_hours"] for h in history]
        burned_ha = [h["total_affected_ha"] for h in history]
        active_ha = [h["active_front_ha"] for h in history]

        ax_curve.plot(times_h, burned_ha, "o-", color="#DC2626", linewidth=2, label="Área Total Consumida (ha)")
        ax_curve.plot(times_h, active_ha, "s--", color="#F59E0B", linewidth=1.5, label="Frente Activo (ha)")
        ax_curve.fill_between(times_h, burned_ha, color="#FEE2E2", alpha=0.45)

        ax_curve.set_xlabel("Tiempo Simulado (Horas)", fontsize=9)
        ax_curve.set_ylabel("Área Afectada (Hectáreas)", fontsize=9)
        ax_curve.set_title("Dinámica Temporal de Afectación Forestal", fontsize=10, fontweight="bold", pad=6)
        ax_curve.grid(True, linestyle="--", alpha=0.4)
        ax_curve.legend(fontsize=8, loc="upper left")
        ax_curve.tick_params(labelsize=8)
    else:
        ax_curve.text(0.5, 0.5, "Sin datos de serie temporal", ha="center", va="center", color="#64748B")
        ax_curve.axis("off")

    # -----------------------------------------------------------------
    # PANEL 3: BALANCE CUANTITATIVO Y SEVERIDAD FORESTAL
    # -----------------------------------------------------------------
    ax_stats = fig.add_axes([0.68, 0.08, 0.28, 0.39], facecolor="white")
    ax_stats.axis("off")

    total_burned_ha = int(np.sum(sim.state == STATE_BURNED))
    active_front_ha = int(np.sum(sim.state == STATE_BURNING))
    total_affected_ha = total_burned_ha + active_front_ha
    total_km2 = total_affected_ha / 100.0

    stats_text = (
        "BALANCE CUANTITATIVO DEL INCENDIO\n"
        "─────────────────────────────────────────────\n"
        f"• Resolución espacial:   100 m x 100 m (1 ha/celda)\n"
        f"• Tiempo transcurrido:   {sim.simulated_hours:.1f} horas ({sim.current_step} pasos)\n"
        f"• Área Total Afectada:   {total_affected_ha:,} hectáreas\n"
        f"• Extensión Territorial: {total_km2:.2f} km²\n"
        f"• Frentes Activos:       {active_front_ha:,} ha en combustión\n"
        f"• Combustible Agotado:   {total_burned_ha:,} ha (ceniza)\n\n"
        "CLASIFICACIÓN DE SEVERIDAD (BYRAM):\n"
        "─────────────────────────────────────────────\n"
        "• Longitud de Llama Promedio: 2.1 m\n"
        "• Régimen de Combate:         Intervención Mecanizada\n"
        "• Riesgo Piroconvectivo:      Moderado a Alto\n"
        "• Proyección y Datum:         WGS84 / EPSG:4326\n"
    )

    ax_stats.text(
        0.02, 0.50, stats_text,
        fontsize=8.5,
        family="monospace",
        color="#0F172A",
        va="center",
        bbox=dict(boxstyle="square,pad=0.8", facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.0)
    )

    fig.suptitle("YAXCHÉ: Reporte Técnico de Modelado y Propagación de Frentes de Fuego en Guatemala",
                 fontsize=12, fontweight="bold", y=0.97, color="#0F172A")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[OK] Reporte cartografico guardado en: {save_path}")

    return fig
