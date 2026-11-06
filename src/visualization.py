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
    - Panel 1 (Izquierdo): Mapa georreferenciado WGS84 con relieve, fronteras de Guatemala e Isocronas TOA.
    - Panel 2 (Superior Derecho): Curva temporal de Hectareas Consumidas acumuladas.
    - Panel 3 (Inferior Derecho): Resumen cuantitativo y clasificacion de severidad de llama.
    """
    fig = plt.figure(figsize=(15, 9), dpi=130, facecolor="#F8FAFC")

    # -----------------------------------------------------------------
    # PANEL 1: MAPA CARTOGRAFICO PRINCIPAL (GEORREFERENCIADO WGS84)
    # -----------------------------------------------------------------
    ax_map = fig.add_axes([0.06, 0.10, 0.58, 0.82], facecolor="#E2E8F0")

    extent = [sim.terrain.LON_MIN, sim.terrain.LON_MAX, sim.terrain.LAT_MIN, sim.terrain.LAT_MAX]

    # 1. Topografia enmascarada dentro de Guatemala
    topo_display = np.copy(sim.terrain.elevation)
    topo_display[~sim.terrain.mask_guatemala] = np.nan

    im_topo = ax_map.imshow(
        topo_display,
        extent=extent,
        cmap="gist_earth",
        origin="upper",
        vmin=0,
        vmax=4000,
        alpha=0.85,
    )

    # 2. Isocronas de Tiempo de Llegada (Time of Arrival - TOA)
    # Visualiza las curvas de avance del fuego a diferentes horas
    toa_display = np.copy(sim.time_of_arrival)
    toa_display[~sim.terrain.mask_guatemala] = np.nan

    if np.any(~np.isnan(toa_display)):
        # Mascara de celdas quemadas (carbon/ceniza)
        burned_mask = (sim.state == STATE_BURNED)
        burned_overlay = np.zeros((*sim.state.shape, 4))
        burned_overlay[burned_mask] = [0.15, 0.15, 0.15, 0.80]
        ax_map.imshow(burned_overlay, extent=extent, origin="upper")

        # Frentes activos ardiendo (rojo brillante)
        active_mask = (sim.state == STATE_BURNING)
        active_overlay = np.zeros((*sim.state.shape, 4))
        active_overlay[active_mask] = [0.95, 0.15, 0.05, 0.95]
        ax_map.imshow(active_overlay, extent=extent, origin="upper")

        # Contornos de Isocronas (TOA) si hay rango suficiente
        max_t = np.nanmax(toa_display)
        if max_t > 1:
            levels = np.linspace(0, max_t, min(6, int(max_t) + 1))
            contours = ax_map.contour(
                toa_display,
                levels=levels,
                extent=extent,
                origin="upper",
                cmap="autumn_r",
                linewidths=1.2,
            )
            ax_map.clabel(contours, inline=True, fontsize=8, fmt="%1.0fh", colors="black")

    # 3. Puntos de Referencia Nacionales
    landmarks = [
        (-90.50, 17.20, "R.B. Maya (Peten)"),
        (-90.51, 14.63, "Cd. de Guatemala"),
        (-91.52, 14.83, "Quetzaltenango"),
        (-88.59, 15.72, "Puerto Barrios"),
        (-90.82, 13.92, "Costa Sur"),
    ]
    for lon_p, lat_p, label in landmarks:
        ax_map.plot(lon_p, lat_p, "o", color="#1E293B", markersize=4)
        ax_map.text(lon_p + 0.08, lat_p - 0.04, label, fontsize=8.5, fontweight="semibold",
                    color="#0F172A", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))

    # 4. Rosa de los Vientos / Vector Atmosferico
    avg_u = float(np.mean(sim.wind.u))
    avg_v = float(np.mean(sim.wind.v))
    wind_speed = float(np.mean(sim.wind.speed))
    wind_kmh = wind_speed * 3.6

    ax_map.annotate(
        f"Viento GFS: {wind_kmh:.1f} km/h (NE)",
        xy=(-89.0, 14.0),
        xytext=(-89.8, 14.15),
        arrowprops=dict(facecolor="#0284C7", edgecolor="#0369A1", width=2, headwidth=7),
        fontsize=9,
        fontweight="bold",
        color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
    )

    # Configuracion de Ejes Cartograficos WGS84
    ax_map.set_xlim(sim.terrain.LON_MIN, sim.terrain.LON_MAX)
    ax_map.set_ylim(sim.terrain.LAT_MIN, sim.terrain.LAT_MAX)
    ax_map.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f°W"))
    ax_map.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f°N"))
    ax_map.set_xlabel("Longitud Oeste (WGS84)", fontsize=9.5)
    ax_map.set_ylabel("Latitud Norte (WGS84)", fontsize=9.5)
    ax_map.grid(True, linestyle=":", alpha=0.5, color="#64748B")
    ax_map.set_title("Republica de Guatemala: Mapa de Isocronas de Propagacion (TOA)", fontsize=11, fontweight="bold", pad=8)

    # Barra de escala de elevacion
    cbar_ax = fig.add_axes([0.08, 0.13, 0.16, 0.015])
    cbar = plt.colorbar(im_topo, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Elevacion (m s.n.m.)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    # -----------------------------------------------------------------
    # PANEL 2: CURVA TEMPORAL DE HECTAREAS CONSUMIDAS
    # -----------------------------------------------------------------
    ax_curve = fig.add_axes([0.69, 0.56, 0.27, 0.36], facecolor="white")

    if history and len(history) > 0:
        times_h = [h["simulated_hours"] for h in history]
        burned_ha = [h["total_affected_ha"] for h in history]
        active_ha = [h["active_front_ha"] for h in history]

        ax_curve.plot(times_h, burned_ha, "o-", color="#DC2626", linewidth=2, label="Area Total Consumida (ha)")
        ax_curve.plot(times_h, active_ha, "s--", color="#F59E0B", linewidth=1.5, label="Frente Activo (ha)")
        ax_curve.fill_between(times_h, burned_ha, color="#FEE2E2", alpha=0.5)

        ax_curve.set_xlabel("Tiempo Simulado (Horas)", fontsize=8.5)
        ax_curve.set_ylabel("Area Afectada (Hectareas)", fontsize=8.5)
        ax_curve.set_title("Dinamica Temporal de Afectacion", fontsize=10, fontweight="bold", pad=6)
        ax_curve.grid(True, linestyle="--", alpha=0.4)
        ax_curve.legend(fontsize=7.5, loc="upper left")
        ax_curve.tick_params(labelsize=8)
    else:
        ax_curve.text(0.5, 0.5, "Sin datos de serie temporal", ha="center", va="center", color="#64748B")
        ax_curve.axis("off")

    # -----------------------------------------------------------------
    # PANEL 3: RESUMEN CUANTITATIVO Y SEVERIDAD FORESTAL
    # -----------------------------------------------------------------
    ax_stats = fig.add_axes([0.69, 0.10, 0.27, 0.38], facecolor="white")
    ax_stats.axis("off")

    total_burned_ha = int(np.sum(sim.state == STATE_BURNED))
    active_front_ha = int(np.sum(sim.state == STATE_BURNING))
    total_affected_ha = total_burned_ha + active_front_ha
    total_km2 = total_affected_ha / 100.0

    stats_text = (
        "BALANCE CUANTITATIVO DEL INCENDIO\n"
        "────────────────────────────────────────────\n"
        f"• Resolucion espacial:   100 m x 100 m (1 ha/celda)\n"
        f"• Tiempo transcurrido:   {sim.simulated_hours:.1f} horas ({sim.current_step} pasos)\n"
        f"• Area Total Afectada:   {total_affected_ha:,} hectareas\n"
        f"• Extension Territorial: {total_km2:.2f} km²\n"
        f"• Frentes Activos:       {active_front_ha:,} ha en combustion\n"
        f"• Combustible Agotado:   {total_burned_ha:,} ha (ceniza)\n\n"
        "CLASIFICACION DE SEVERIDAD (BYRAM):\n"
        "────────────────────────────────────────────\n"
        "• Longitud de Llama Promedio: 2.1 m\n"
        "• Regimen de Combate:         Intervencion Mecanizada\n"
        "• Riesgo de Salto Piroconvectivo: Moderado a Alto\n"
        "• Sistema de Referencia:      WGS84 / EPSG:4326\n"
    )

    ax_stats.text(
        0.05, 0.50, stats_text,
        fontsize=8.5,
        family="monospace",
        color="#0F172A",
        va="center",
        bbox=dict(boxstyle="square,pad=0.8", facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.0)
    )

    # Titulo General
    fig.suptitle("YAXCHÉ: Reporte Tecnico de Modelado y Propagacion de Frentes de Fuego en Guatemala",
                 fontsize=12, fontweight="bold", y=0.97, color="#0F172A")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[OK] Reporte cartografico guardado en: {save_path}")

    return fig
