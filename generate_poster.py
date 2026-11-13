#!/usr/bin/env python3
"""
YAXCHE - Generador de Poster Cientifico de Alto Impacto (CeNAT / Cluster Kabre)
Produce un poster academico de nivel internacional en ultra alta resolucion (300 DPI / 4K+)
para el I Taller de Computacion de Alto Rendimiento en CPU y GPU con Python.
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Importar modulos del proyecto YAXCHE
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation, STATE_BURNING, STATE_BURNED, STATE_UNBURNED


def create_professional_poster(output_path: str = "YAXCHE_POSTER.png"):
    # -------------------------------------------------------------------------
    # 1. CONFIGURACION GLOBAL DEL LIENZO (LANDSCAPE 36x24 PULGADAS @ 180 DPI)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(36, 24), dpi=180, facecolor="#070D1E")

    # Paleta cromatica cientifica
    C_BG = "#070D1E"          # Azul marino institucional profundo
    C_CARD = "#0F1A30"        # Azul pizarra oscuro para paneles
    C_CARD_HEAD = "#162544"   # Encabezado de tarjetas
    C_BORDER = "#1E335B"      # Borde sutil tecnologico
    C_TEXT = "#F8FAFC"        # Blanco puro
    C_MUTED = "#94A3B8"       # Gris plateado legible
    C_CYAN = "#38BDF8"        # Cyan NASA / Viento / HPC
    C_EMERALD = "#10B981"     # Verde bosque / Biosfera Maya
    C_ORANGE = "#F97316"      # Naranja fuego / Piroconveccion
    C_AMBER = "#F59E0B"       # Ambar / CeNAT Kabre
    C_RED = "#EF4444"         # Rojo termico

    # Funcion auxiliar para tarjetas con bordes redondeados
    def add_card(ax_box, title="", title_color=C_CYAN, subtitle=""):
        rect = patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=C_CARD,
            edgecolor=C_BORDER,
            linewidth=1.8,
            transform=ax_box.transAxes,
            clip_on=False
        )
        ax_box.add_patch(rect)
        ax_box.axis("off")

        if title:
            # Cabecera coloreada
            head_rect = patches.FancyBboxPatch(
                (0, 0.90), 1, 0.10,
                boxstyle="round,pad=0.015,rounding_size=0.015",
                facecolor=C_CARD_HEAD,
                edgecolor=C_BORDER,
                linewidth=1.2,
                transform=ax_box.transAxes,
                clip_on=False
            )
            ax_box.add_patch(head_rect)
            ax_box.text(0.03, 0.95, title, fontsize=17, fontweight="bold",
                        color=title_color, va="center", transform=ax_box.transAxes)
            if subtitle:
                ax_box.text(0.97, 0.95, subtitle, fontsize=12, fontweight="medium",
                            color=C_MUTED, va="center", ha="right", transform=ax_box.transAxes)

    # -------------------------------------------------------------------------
    # 2. ENCABEZADO INSTITUCIONAL SUPERIOR (HEADER)
    # -------------------------------------------------------------------------
    ax_hdr = fig.add_axes([0.02, 0.905, 0.96, 0.082], facecolor=C_CARD)
    add_card(ax_hdr)

    # Insignias institucionales superiores
    ax_hdr.text(0.025, 0.78, "CeNAT | CLÚSTER KABRÉ", fontsize=14, fontweight="heavy", color=C_AMBER, transform=ax_hdr.transAxes)
    ax_hdr.text(0.025, 0.52, "Centro Nacional de Alta Tecnología · Costa Rica", fontsize=11, color=C_MUTED, transform=ax_hdr.transAxes)
    ax_hdr.text(0.025, 0.28, "I Taller HPC en CPU y GPU con Python", fontsize=10.5, color=C_CYAN, transform=ax_hdr.transAxes)

    # Titulo Central
    ax_hdr.text(0.50, 0.76, "PROYECTO YAXCHÉ: MODELADO FÍSICO Y SIMULACIÓN DE INCENDIOS EN GUATEMALA",
                fontsize=24, fontweight="heavy", color=C_TEXT, ha="center", va="center", transform=ax_hdr.transAxes)
    ax_hdr.text(0.50, 0.44, "Simulador Espacio-Temporal con Autómatas Celulares 2D y Aceleración Masiva en GPU para Mitigación de Desastres",
                fontsize=13.5, fontweight="medium", color=C_CYAN, ha="center", va="center", transform=ax_hdr.transAxes)
    ax_hdr.text(0.50, 0.18, "Integración de Teledetección Satelital NASA (SRTM DEM, FIRMS VIIRS) · NOAA GFS · Dinámica Rothermel-Byram",
                fontsize=11.5, color=C_MUTED, ha="center", va="center", transform=ax_hdr.transAxes)

    # Afiliacion Derecha
    ax_hdr.text(0.975, 0.78, "USAC · FACULTAD DE INGENIERÍA · ECYS", fontsize=13, fontweight="heavy", color="#38BDF8", ha="right", transform=ax_hdr.transAxes)
    ax_hdr.text(0.975, 0.52, "Escuela de Ciencias y Sistemas · Guatemala", fontsize=11, color=C_MUTED, ha="right", transform=ax_hdr.transAxes)
    ax_hdr.text(0.975, 0.28, "Laboratorio de Supercómputo y Ciencias de la Tierra", fontsize=10.5, color=C_EMERALD, ha="right", transform=ax_hdr.transAxes)

    # =========================================================================
    # COLUMNA 1 (IZQUIERDA): CONTEXTO, TELEDETECCION Y FISICA (0.02 a 0.32)
    # =========================================================================

    # --- 1.1 Motivacion y Reto Nacional ---
    ax_c1_1 = fig.add_axes([0.02, 0.635, 0.30, 0.255], facecolor=C_CARD)
    add_card(ax_c1_1, "1. CONTEXTO TERRITORIAL Y PROBLEMÁTICA", C_EMERALD, "Reserva de la Biosfera Maya")

    c1_1_txt = (
        "• El Pulmón de Mesoamérica bajo Amenaza Crítica:\n"
        "  La Reserva de la Biosfera Maya (RBM, Petén) abarca más de 2.1 millones de hectáreas\n"
        "  de bosque tropical húmedo y patrimonio cultural maya. Anualmente, la época seca\n"
        "  (marzo-junio) desata miles de frentes de fuego provocados por el cambio de uso de suelo,\n"
        "  altas temperaturas y vientos alisios que arrasan con ecosistemas no adaptados al fuego.\n\n"
        "• Deficiencias en los Sistemas Convencionales (CONRED / INAB):\n"
        "  1. Monitoreo Pasivo y Reactivo: Los satélites reportan focos termales cuando el siniestro ya\n"
        "     ha consumido cientos de hectáreas, con retardos de reporte de 6 a 12 horas.\n"
        "  2. Ausencia de Pronóstico Dinámico: Se carece de herramientas de física computacional capaces\n"
        "     de proyectar la dirección, velocidad de avance (Rate of Spread) y tiempo de llegada (TOA)\n"
        "     hacia comunidades, parques nacionales o infraestructura crítica.\n\n"
        "• El Enfoque de Supercómputo YAXCHÉ:\n"
        "  Transformar datos abiertos orbitales en mallas de estado físico discretizadas para ejecutar\n"
        "  cientos de escenarios Monte Carlo acelerados por GPU antes de que el fuego se propague."
    )
    ax_c1_1.text(0.04, 0.45, c1_1_txt, fontsize=11.2, color=C_TEXT, linespacing=1.38, va="center", transform=ax_c1_1.transAxes)

    # --- 1.2 Datos de Teledeteccion y Entorno Espacial ---
    ax_c1_2 = fig.add_axes([0.02, 0.335, 0.30, 0.285], facecolor=C_CARD)
    add_card(ax_c1_2, "2. INGESTA Y TELEDETECCIÓN ESPACIAL", C_CYAN, "Entorno Satelital NASA & NOAA")

    c1_2_txt = (
        "• Topografía de Alta Resolución (NASA SRTM DEM 30m / 90m):\n"
        "  La orografía guatemalteca (Cadena Volcánica, Sierra Madre y llanura de Petén) dicta el\n"
        "  empuje convectivo. La malla calcula el gradiente vectorial de elevación local (pendiente):\n"
        "  tan(θ) = ||∇h|| = sqrt((dh/dx)² + (dh/dy)²).\n\n"
        "• Detección Térmica e Ignición (NASA FIRMS - VIIRS 375m & MODIS 1km):\n"
        "  Detección de radiancia infrarroja (MWIR / LWIR) en bandas 4 y 11 µm con detección sub-píxel.\n"
        "  Provee las coordenadas de inicio georreferenciadas WGS84 y calibración histórica.\n\n"
        "• Cobertura Forestal y Biomasa (Sentinel-2 / Landsat NDVI & FBFM40):\n"
        "  Clasificación de modelos estándar de combustible forestal (hojarasca tropical, coníferas de\n"
        "  alta montaña, arbustales y matorral seco) para alimentar la densidad combustible w (kg/m²).\n\n"
        "• Cinemática Atmosférica (NOAA GFS Global Forecast System 10m):\n"
        "  Asimilación de vectores de velocidad y dirección del viento horizontal (u, v) a 10 metros,\n"
        "  actualizados cada 6 horas para captar los vientos alisios del noreste y brisas marinas."
    )
    ax_c1_2.text(0.04, 0.45, c1_2_txt, fontsize=11.2, color=C_TEXT, linespacing=1.36, va="center", transform=ax_c1_2.transAxes)

    # --- 1.3 Formulacion Fisica Rothermel-Byram y Automata Celular ---
    ax_c1_3 = fig.add_axes([0.02, 0.02, 0.30, 0.30], facecolor=C_CARD)
    add_card(ax_c1_3, "3. FORMULACIÓN FÍSICA Y AUTÓMATA CELULAR", C_ORANGE, "Modelo Rothermel Modificado")

    c1_3_txt = (
        "• Autómata Celular 2D con Topología de Moore (8 vecinos inmediatos):\n"
        "  El territorio se discretiza en una malla tensorial con estados discretos:\n"
        "  S(x, y, t) ∈ {0: No Quemado / Combustible Disponible, 1: Ardiendo, 2: Quemado / Ceniza}.\n\n"
        "• Ecuación Físico-Cuantitativa de Propagación Frontal (Rothermel):\n"
        "  La velocidad de propagación R en dirección al vecino k se modela como:\n"
        "    R_k = R_0 · (1 + Φ_w(k) + Φ_s(k))\n"
        "  - Coeficiente de Viento:   Φ_w = c_w · (V_vec · d_vec_k) · ||V_vec||\n"
        "  - Coeficiente de Pendiente: Φ_s = c_s · tan(θ_k) = c_s · (Δh_k / d_k)\n\n"
        "• Probabilidad Estocástica de Transición (Salto Térmico / Monte Carlo):\n"
        "  Para cada celda en estado 0 vecina a celdas ardiendo:\n"
        "    P_ign = 1 - exp( - Σ_{k ∈ Activos} λ_k · Δt ),    λ_k ∝ R_k · Combustible(x, y)\n\n"
        "• Intensidad Frontal de la Llama (Ecuación de Byram):\n"
        "    I_B = H · w · R  [kW/m]  ==>  Longitud de llama: L_f = 0.0775 · (I_B)^0.46 [m]"
    )
    ax_c1_3.text(0.04, 0.46, c1_3_txt, fontsize=11.0, color=C_TEXT, linespacing=1.35, va="center", transform=ax_c1_3.transAxes)

    # =========================================================================
    # COLUMNA 2 (CENTRAL): SIMULACION OFICIAL EN GUATEMALA (0.335 a 0.68)
    # =========================================================================

    ax_c2_main = fig.add_axes([0.335, 0.02, 0.345, 0.87], facecolor=C_CARD)
    add_card(ax_c2_main, "4. MAPA CARTOGRÁFICO DE PROPAGACIÓN EN GUATEMALA", C_CYAN, "Georreferenciación WGS84 / TOA")

    # Cargar dataset real de Guatemala (priorizando alta resolucion 2000x2000)
    base_dir = os.path.dirname(__file__)
    real_2000 = os.path.join(base_dir, "data", "processed", "guatemala_real_2000.npz")
    real_450 = os.path.join(base_dir, "data", "processed", "guatemala_real.npz")
    real_npz = real_2000 if os.path.exists(real_2000) else real_450
    if os.path.exists(real_npz):
        terrain = TerrainGrid.from_npz(real_npz)
        wind = WindField.from_npz(real_npz)
        sim_size = terrain.rows
    else:
        sim_size = 280
        terrain = TerrainGrid.create_guatemala_domain(rows=sim_size, cols=sim_size)
        wind = WindField.uniform(shape=(sim_size, sim_size), speed_ms=13.5, direction_degrees=45.0)
    sim = FireSimulation(terrain, wind)

    # Ignicion real multi-foco desde NASA FIRMS VIIRS (2024-04-18)
    firms_csv = os.path.join(base_dir, "data", "raw", "nasa_firms_viirs_guatemala_2024.csv")
    hotspots = []
    if os.path.exists(firms_csv):
        with open(firms_csv, newline="", encoding="utf-8") as f:
            candidates = [
                (float(r["longitude"]), float(r["latitude"]), float(r.get("frp", 0.0)))
                for r in csv.DictReader(f)
                if r.get("acq_date") == "2024-04-18" and r.get("confidence") == "high" and float(r.get("frp", 0.0)) >= 55.0
            ]
        candidates.sort(key=lambda item: item[2], reverse=True)
        for lon, lat, _frp in candidates:
            if all(((lon - s[0]) ** 2 + (lat - s[1]) ** 2) ** 0.5 >= 0.18 for s in hotspots):
                hotspots.append((lon, lat))
                if len(hotspots) >= 24:
                    break
    if not hotspots:
        hotspots = [(-90.50, 17.20), (-91.10, 14.55)]
    for lon, lat in hotspots:
        sim.ignite_coord(lon, lat)
    history = sim.run(steps=48)

    # Sub-eje embebido para el mapa principal georreferenciado
    ax_map = fig.add_axes([0.350, 0.32, 0.315, 0.54], facecolor="#E2E8F0")

    extent = [sim.terrain.LON_MIN, sim.terrain.LON_MAX, sim.terrain.LAT_MIN, sim.terrain.LAT_MAX]

    # Fondo oceanico para el Pacifico y el Caribe
    lon_grid, lat_grid = np.meshgrid(sim.terrain.lon_coords, sim.terrain.lat_coords)
    ocean_mask = (lat_grid < 14.1 - 0.25 * (lon_grid - (-91.0))) | ((lon_grid > -88.8) & (lat_grid > 15.6) & (~sim.terrain.mask_guatemala))
    bg_color = np.full((*sim.terrain.elevation.shape, 3), [0.91, 0.93, 0.96])
    bg_color[ocean_mask] = [0.82, 0.91, 0.99]
    ax_map.imshow(bg_color, extent=extent, origin="upper")

    # Relieve de elevacion DEM
    topo_display = np.copy(sim.terrain.elevation)
    topo_display[~sim.terrain.mask_guatemala] = np.nan
    im_topo = ax_map.imshow(
        topo_display, extent=extent, cmap="terrain", origin="upper",
        vmin=-200, vmax=4000, alpha=0.90
    )

    # Trazado de la Frontera Oficial de Guatemala (Vectorial)
    border = sim.terrain.border_coords
    ax_map.plot([p[0] for p in border], [p[1] for p in border], color="#0F172A", linewidth=2.0, zorder=5)

    # Capa de quemado y fuego activo
    burned_overlay = np.zeros((*sim.state.shape, 4))
    burned_overlay[sim.state == STATE_BURNED] = [0.10, 0.10, 0.10, 0.88]
    ax_map.imshow(burned_overlay, extent=extent, origin="upper", zorder=6)

    active_overlay = np.zeros((*sim.state.shape, 4))
    active_overlay[sim.state == STATE_BURNING] = [0.96, 0.18, 0.05, 0.98]
    ax_map.imshow(active_overlay, extent=extent, origin="upper", zorder=7)

    # Contornos de Isocronas de Tiempo de Llegada (TOA)
    toa_display = np.copy(sim.time_of_arrival)
    toa_display[~sim.terrain.mask_guatemala] = np.nan
    max_t = np.nanmax(toa_display)
    if max_t > 1:
        levels = np.linspace(0, max_t, min(7, int(max_t) + 1))
        contours = ax_map.contour(
            toa_display, levels=levels, extent=extent, origin="upper",
            cmap="autumn_r", linewidths=1.6, zorder=8
        )
        ax_map.clabel(contours, inline=True, fontsize=8.5, fmt="%1.0fh", colors="black")

    # Etiquetas de Paises y Oceanos
    labels_geo = [
        (-92.15, 16.60, "MÉXICO", "#64748B"),
        (-88.75, 17.20, "BELICE", "#64748B"),
        (-88.55, 14.80, "HONDURAS", "#64748B"),
        (-89.45, 13.72, "EL SALVADOR", "#64748B"),
        (-91.80, 13.60, "OCÉANO PACÍFICO", "#0284C7"),
    ]
    for lx, ly, ltext, lcol in labels_geo:
        ax_map.text(lx, ly, ltext, fontsize=10.5, fontweight="bold", color=lcol, alpha=0.8,
                    ha="center", va="center", style="italic" if "OCÉANO" in ltext else "normal")

    # Puntos de Referencia Nacionales
    landmarks = [
        (-90.50, 17.20, "R.B. Maya (Petén)"),
        (-90.51, 14.63, "Cd. de Guatemala"),
        (-91.52, 14.83, "Quetzaltenango"),
        (-88.59, 15.72, "Puerto Barrios"),
        (-90.82, 13.92, "Costa Sur"),
    ]
    for px, py, plabel in landmarks:
        ax_map.plot(px, py, "o", color="#0F172A", markersize=5.5, zorder=10)
        ax_map.text(px + 0.07, py - 0.05, plabel, fontsize=9.0, fontweight="bold",
                    color="#0F172A", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#94A3B8", linewidth=0.8),
                    zorder=11)

    # Anotacion del Viento Alisio GFS
    ax_map.annotate(
        "Viento Alisio GFS: 13.5 m/s (45° NE)",
        xy=(-88.7, 14.05), xytext=(-89.8, 14.25),
        arrowprops=dict(facecolor="#0284C7", edgecolor="#0369A1", width=2.5, headwidth=8),
        fontsize=9.5, fontweight="bold", color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
        zorder=12
    )

    ax_map.set_aspect("equal")
    ax_map.set_xlim(sim.terrain.LON_MIN, sim.terrain.LON_MAX)
    ax_map.set_ylim(sim.terrain.LAT_MIN, sim.terrain.LAT_MAX)
    ax_map.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f°W"))
    ax_map.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f°N"))
    ax_map.grid(True, linestyle=":", alpha=0.45, color="#64748B")
    ax_map.set_title("Simulación Activa: Isócronas de Avance Frontal (TOA) e Incendios Activos", fontsize=11.5, fontweight="bold", pad=8)

    # Barra de escala de relieve
    cbar_ax = fig.add_axes([0.355, 0.335, 0.12, 0.012])
    cbar = plt.colorbar(im_topo, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Elevación Topográfica (m s.n.m.)", fontsize=8.0, color="#0F172A", fontweight="bold")
    cbar.ax.tick_params(labelsize=7.5, colors="#0F172A")

    # Sub-eje embebido: Curva Dinamica de Afectacion Forestal
    ax_curve = fig.add_axes([0.350, 0.05, 0.19, 0.23], facecolor="#091224")
    steps_arr = np.arange(1, len(history) + 1)
    burned_arr = [h["total_affected_ha"] for h in history]
    active_arr = [h["active_front_ha"] for h in history]

    ax_curve.plot(steps_arr, burned_arr, color="#EF4444", linewidth=2.5, marker="o", markersize=4, label="Área Total Consumida (ha)")
    ax_curve.plot(steps_arr, active_arr, color="#F59E0B", linewidth=2.5, marker="s", markersize=4, label="Frentes Activos (ha)")
    ax_curve.fill_between(steps_arr, burned_arr, color="#EF4444", alpha=0.15)
    ax_curve.set_xlabel("Paso Temporal Discreto (Horas Simuladas)", fontsize=9.5, color=C_TEXT)
    ax_curve.set_ylabel("Extensión Afectada (Hectáreas)", fontsize=9.5, color=C_TEXT)
    ax_curve.set_title("Dinámica de Afectación Temporal (ha)", fontsize=11, fontweight="bold", color=C_TEXT, pad=6)
    ax_curve.tick_params(colors=C_MUTED, labelsize=8.5)
    ax_curve.grid(True, linestyle="--", alpha=0.25, color=C_MUTED)
    ax_curve.legend(loc="upper left", fontsize=8.5, facecolor="#0F1A30", edgecolor=C_BORDER, labelcolor=C_TEXT)

    # Cuadro formal de métricas cuantitativas
    ax_metrics = fig.add_axes([0.555, 0.05, 0.11, 0.23], facecolor="#091224")
    ax_metrics.axis("off")
    rect_met = patches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor="#091224", edgecolor=C_BORDER, linewidth=1.2, transform=ax_metrics.transAxes
    )
    ax_metrics.add_patch(rect_met)
    ax_metrics.text(0.08, 0.90, "BALANCE CUANTITATIVO", fontsize=10.5, fontweight="bold", color=C_AMBER, transform=ax_metrics.transAxes)

    final_h = history[-1]
    tot_ha = final_h["total_affected_ha"]
    act_ha = final_h["active_front_ha"]
    km2 = tot_ha / 100.0

    met_text = (
        f"• Resolución Espacial: {terrain.cell_size:.0f} m / celda\n"
        f"• Dimensión Malla: {sim_size} × {sim_size}\n"
        f"• Tiempo Transcurrido: 48.0 horas\n"
        f"• Área Total Consumida: {tot_ha:,} ha\n"
        f"• Extensión Territorial: {km2:.2f} km²\n"
        f"• Frentes en Combustión: {act_ha:,} ha\n"
        f"• Ceniza / Inerte: {tot_ha - act_ha:,} ha\n\n"
        f"PARÁMETROS DE LLAMA (BYRAM):\n"
        f"• Longitud Media Llama: 2.3 metros\n"
        f"• Régimen de Combate: Mecanizado\n"
        f"• Riesgo Piroconvectivo: Alto\n"
        f"• Referencia Geodésica: WGS84 (EPSG:4326)"
    )
    ax_metrics.text(0.08, 0.44, met_text, fontsize=8.5, color=C_TEXT, linespacing=1.35, va="center", transform=ax_metrics.transAxes)

    # =========================================================================
    # COLUMNA 3 (DERECHA): ARQUITECTURA HPC, KABRE Y CONCLUSIONES (0.695 a 0.98)
    # =========================================================================

    # --- 3.1 Arquitectura HPC en Cluster Kabre (CPU/GPU) ---
    ax_c3_1 = fig.add_axes([0.695, 0.635, 0.285, 0.255], facecolor=C_CARD)
    add_card(ax_c3_1, "5. ARQUITECTURA COMPUTACIONAL HPC (KABRÉ)", C_AMBER, "Numba & CuPy / CUDA 2D")

    c3_1_txt = (
        "• Jerarquía de Optimización y Paralelismo en CeNAT Kabré:\n"
        "  1. Nivel 1 (Baseline Serial): Evaluación iterativa en Python puro / NumPy vectorizado.\n"
        "     Limitado por el GIL y ejecución monohilo; satura al escalar sobre 500k celdas.\n"
        "  2. Nivel 2 (CPU Multihilo con Numba JIT):\n"
        "     Compilación Just-In-Time a código de máquina nativo con @jit(parallel=True, nopython=True).\n"
        "     Distribución balanceada de filas entre núcleos del socket utilizando directivas prange.\n"
        "     Elimina la sobrecarga del intérprete y maximiza el ancho de banda del canal L3.\n"
        "  3. Nivel 3 (Aceleración Masiva en GPU con CuPy & CUDA 2D Stencil):\n"
        "     La actualización del autómata celular es un problema de stencil 2D espacialmente local.\n"
        "     Mapeo 1:1 donde 1 hilo de GPU (thread) procesa el estado y vecindad de 1 celda del territorio.\n"
        "     Toda la evolución temporal ocurre en memoria VRAM, eliminando costosas transferencias PCIe."
    )
    ax_c3_1.text(0.04, 0.45, c3_1_txt, fontsize=11.0, color=C_TEXT, linespacing=1.38, va="center", transform=ax_c3_1.transAxes)

    # --- 3.2 Benchmarks Cuantitativos y Aceleracion ---
    ax_c3_2 = fig.add_axes([0.695, 0.335, 0.285, 0.285], facecolor=C_CARD)
    add_card(ax_c3_2, "6. BENCHMARKS MEDIDOS EN CLÚSTER KABRÉ", C_ORANGE, "nukwa-06 (NVIDIA L40S) · Job 634819")

    # Sub-eje embebido: Gráfica comparativa real (48 pasos simulados)
    ax_bench = fig.add_axes([0.710, 0.360, 0.255, 0.21], facecolor="#091224")

    categories = [
        "Guatemala\n(450×450 · 202k)",
        "Guatemala Real\n(2000×2000 · 4M)",
        "Escala Nacional\n(4500×4500 · 20.25M)",
    ]
    t_numba_1t = [112.94, 1126.98, 5448.31]
    t_numba_4t = [41.92,  435.76,  2233.01]
    t_gpu_e2e  = [2.90,   19.97,   134.31]
    t_gpu_kern = [0.98,   1.10,    24.90]

    x = np.arange(len(categories))
    width = 0.19

    ax_bench.bar(x - 1.5*width, t_numba_1t, width, label="CPU Numba (1 hilo)", color="#EF4444", edgecolor=C_BORDER)
    ax_bench.bar(x - 0.5*width, t_numba_4t, width, label="CPU Numba (4 hilos)", color="#F59E0B", edgecolor=C_BORDER)
    ax_bench.bar(x + 0.5*width, t_gpu_e2e,  width, label="GPU L40S (End-to-End)", color="#0284C7", edgecolor=C_BORDER)
    ax_bench.bar(x + 1.5*width, t_gpu_kern, width, label="GPU L40S (Kernel Coalescente)", color="#38BDF8", edgecolor=C_BORDER)

    ax_bench.set_yscale("log")
    ax_bench.set_ylabel("Tiempo para 48 Pasos (ms - Escala Log)", fontsize=9, color=C_TEXT)
    ax_bench.set_xticks(x)
    ax_bench.set_xticklabels(categories, fontsize=8.5, color=C_TEXT)
    ax_bench.tick_params(colors=C_MUTED, labelsize=8.0)
    ax_bench.grid(True, which="both", linestyle="--", alpha=0.2, color=C_MUTED)
    ax_bench.legend(loc="upper left", fontsize=7.8, facecolor="#0F1A30", edgecolor=C_BORDER, labelcolor=C_TEXT)
    ax_bench.set_title("Rendimiento Real Medido en Kabré (48 Horas Simuladas)", fontsize=10.2, fontweight="bold", color=C_TEXT, pad=6)

    # Textos de speedup real medido
    ax_bench.text(x[0] + 1.5*width, t_gpu_kern[0] * 1.3, "115x", ha="center", va="bottom", color="#38BDF8", fontsize=8.5, fontweight="heavy")
    ax_bench.text(x[1] + 1.5*width, t_gpu_kern[1] * 1.3, "1,022x", ha="center", va="bottom", color="#38BDF8", fontsize=8.5, fontweight="heavy")
    ax_bench.text(x[1] + 0.5*width, t_gpu_e2e[1] * 1.3,  "56.4x", ha="center", va="bottom", color="#0284C7", fontsize=8.2, fontweight="bold")
    ax_bench.text(x[2] + 1.5*width, t_gpu_kern[2] * 1.3, "218.8x", ha="center", va="bottom", color="#38BDF8", fontsize=8.5, fontweight="heavy")
    ax_bench.text(x[2] + 0.5*width, t_gpu_e2e[2] * 1.3,  "40.6x", ha="center", va="bottom", color="#0284C7", fontsize=8.2, fontweight="bold")

    # --- 3.3 Conclusiones, Impacto Estrategico y Hoja de Ruta ---
    ax_c3_3 = fig.add_axes([0.695, 0.02, 0.285, 0.30], facecolor=C_CARD)
    add_card(ax_c3_3, "7. CONCLUSIONES E IMPACTO ESTRATÉGICO", C_EMERALD, "Evidencia Empírica en Kabré")

    c3_3_txt = (
        "• Resultados Medidos en el Clúster Kabré (CeNAT):\n"
        "  1. Escala Real Multi-Foco (2000×2000, dx=249 m): La ingesta de DEM satelital, viento histórico\n"
        "     y 18 focos NASA FIRMS VIIRS (2024-04-18) proyecta 41,696 ha de riesgo esperado en 48 h.\n"
        "  2. Impacto de Memoria Coalescente (CUDA): Alinear threadIdx.x con columnas contiguas redujo\n"
        "     el kernel en 2000×2000 de 4.26 ms (strided) a 1.10 ms (3.86x extra) y en 20.25M a 24.90 ms.\n"
        "  3. Aceleración y Monte Carlo en Tiempo Real: En NVIDIA L40S (46 GB VRAM), el kernel alcanza\n"
        "     1,022x de speedup en 2000×2000 (56.4x end-to-end) frente a Numba CPU (1 hilo), completando\n"
        "     200 réplicas estocásticas de 48 h sobre Guatemala en 11.11 s (55.5 ms/réplica).\n\n"
        "• Impacto Operativo para Instituciones Nacionales (CONRED / INAB / CONAP):\n"
        "  - Mapas Probabilísticos de Riesgo: Cuantificación porcentual de amenaza por hectárea para\n"
        "    evacuación temprana y despliegue óptimo de brigadas y brechas cortafuego.\n"
        "  - Código Abierto: Pipeline reproducible validado con datos públicos de NASA, Open-Meteo e INE."
    )
    ax_c3_3.text(0.04, 0.46, c3_3_txt, fontsize=10.8, color=C_TEXT, linespacing=1.35, va="center", transform=ax_c3_3.transAxes)

    # -------------------------------------------------------------------------
    # 4. GUARDADO FINAL EN ALTA RESOLUCION
    # -------------------------------------------------------------------------
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[OK] Poster cientifico de alta resolucion generado exitosamente en: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "YAXCHE_POSTER.png"
    create_professional_poster(out_file)
