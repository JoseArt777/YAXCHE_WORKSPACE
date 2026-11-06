#!/usr/bin/env python3
"""
YAXCHE - Generador de Poster Cientifico e Infografia Tecnica
Genera una imagen en alta resolucion con el resumen completo del proyecto.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.terrain import TerrainGrid
from src.wind import WindField
from src.simulation import FireSimulation, STATE_BURNING, STATE_BURNED


def build_academic_poster(output_path: str = "YAXCHE_POSTER.png"):
    # Configuración de lienzo (Proporción vertical 4:5 estilo poster de conferencia)
    fig = plt.figure(figsize=(16, 20), dpi=180, facecolor="#0B132B")
    
    # Colores temáticos
    C_BG = "#0B132B"         # Azul oscuro profundo
    C_CARD = "#1C2541"       # Azul pizarra para tarjetas
    C_BORDER = "#3A506B"     # Borde sutil
    C_TEXT = "#FFFFFF"       # Blanco principal
    C_MUTED = "#94A3B8"      # Gris texto secundario
    C_ACCENT_BLUE = "#38BDF8"# Azul celeste (NASA / Viento)
    C_ACCENT_GREEN = "#10B981" # Verde esmeralda (Ceiba / Biomasa)
    C_ACCENT_ORANGE = "#F97316"# Naranja fuego
    
    # -------------------------------------------------------------
    # 1. ENCABEZADO PRINCIPAL (HEADER)
    # -------------------------------------------------------------
    header_ax = fig.add_axes([0.05, 0.88, 0.90, 0.09], facecolor=C_CARD)
    header_ax.axis("off")
    rect = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.5, transform=header_ax.transAxes)
    header_ax.add_patch(rect)
    
    header_ax.text(0.5, 0.72, "PROYECTO YAXCHÉ", fontsize=28, fontweight="bold",
                   color=C_TEXT, ha="center", va="center", transform=header_ax.transAxes)
    header_ax.text(0.5, 0.45, "Simulador Espacio-Temporal y Acelerado por GPU para la Mitigación de Incendios en Guatemala",
                   fontsize=14, fontweight="medium", color=C_ACCENT_BLUE, ha="center", va="center", transform=header_ax.transAxes)
    header_ax.text(0.5, 0.20, "Centro Nacional de Alta Tecnologia (CeNAT) | Cluster Kabre | NASA Earth Science Data | ECYS USAC",
                   fontsize=11, color=C_MUTED, ha="center", va="center", transform=header_ax.transAxes)

    # -------------------------------------------------------------
    # 2. PANEL IZQUIERDO SUPERIOR: CONTEXTO E IDENTIDAD
    # -------------------------------------------------------------
    p1_ax = fig.add_axes([0.05, 0.65, 0.43, 0.21], facecolor=C_CARD)
    p1_ax.axis("off")
    p1_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p1_ax.transAxes))
    
    p1_ax.text(0.06, 0.90, "1. IDENTIDAD Y FUNDAMENTO CIENTIFICO", fontsize=13, fontweight="bold", color=C_ACCENT_GREEN, transform=p1_ax.transAxes)
    
    text_p1 = (
        "• Origen Cosmovision Maya:\n"
        "  La Yaxche (Ceiba Pentandra) es el Arbol de la Vida y eje cosmico\n"
        "  que conecta las estrellas en el cielo con las raices en el suelo.\n"
        "  Ademas, es el Arbol Nacional y simbolo de resiliencia de Guatemala.\n\n"
        "• La Metafora Tecnologica:\n"
        "  YAXCHE emula este puente captando datos de observacion desde\n"
        "  la orbita espacial (satelites NASA) para transferirlos a la masa\n"
        "  forestal del territorio nacional y predecir frentes termicos.\n\n"
        "• El Reto en la Reserva de la Biosfera Maya (RBM):\n"
        "  Las herramientas actuales de CONRED/INAB son reactivas (alertas\n"
        "  tardias) o mapas de recurrencia estatica sin pronostico dinamico."
    )
    p1_ax.text(0.06, 0.44, text_p1, fontsize=10.2, color=C_TEXT, linespacing=1.35, va="center", transform=p1_ax.transAxes)

    # -------------------------------------------------------------
    # 3. PANEL DERECHO SUPERIOR: INGESTA DE DATOS ABIERTOS (NASA)
    # -------------------------------------------------------------
    p2_ax = fig.add_axes([0.52, 0.65, 0.43, 0.21], facecolor=C_CARD)
    p2_ax.axis("off")
    p2_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p2_ax.transAxes))
    
    p2_ax.text(0.06, 0.90, "2. INGESTA Y MODELADO ESPACIAL (NASA)", fontsize=13, fontweight="bold", color=C_ACCENT_BLUE, transform=p2_ax.transAxes)
    
    text_p2 = (
        "• Topografia (DEM) - NASA SRTM (30m / 90m):\n"
        "  Matrices de elevacion para computo de pendientes y orientacion.\n\n"
        "• Deteccion Termica - NASA FIRMS (VIIRS & MODIS):\n"
        "  Puntos de calor historicos y recientes para calibrar igniciones.\n\n"
        "• Cobertura y Biomasa - Sentinel-2 / Landsat (NDVI):\n"
        "  Cuantificacion celda por celda de la densidad de combustible vivo.\n\n"
        "• Campos de Viento - NOAA GFS (10m u, v):\n"
        "  Vectores de velocidad y direccion atmosferica en tiempo real.\n\n"
        "• Mapeo Tensorial:\n"
        "  Discretizacion territorial en mallas multivariadas de 100m."
    )
    p2_ax.text(0.06, 0.44, text_p2, fontsize=10.2, color=C_TEXT, linespacing=1.35, va="center", transform=p2_ax.transAxes)

    # -------------------------------------------------------------
    # 4. PANEL IZQUIERDO CENTRAL: FORMULACION MATEMATICA Y CELULAR
    # -------------------------------------------------------------
    p3_ax = fig.add_axes([0.05, 0.36, 0.43, 0.27], facecolor=C_CARD)
    p3_ax.axis("off")
    p3_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p3_ax.transAxes))
    
    p3_ax.text(0.06, 0.92, "3. FISICA Y AUTOMATA CELULAR 2D", fontsize=13, fontweight="bold", color=C_ACCENT_ORANGE, transform=p3_ax.transAxes)
    
    text_p3 = (
        "• Espacio de Estados Discretos:\n"
        "  S = {0: No Quemado,  1: Ardiendo,  2: Quemado/Ceniza}\n\n"
        "• Topologia de Vecindad de Moore:\n"
        "  Evaluacion simultanea de los 8 vecinos inmediatos por celda.\n\n"
        "• Ecuacion Probabilistica de Transicion (Rothermel Modificado):\n"
        "  P(ignicion) = P_base * Factor_Combustible * Factor_Pendiente * Factor_Viento\n\n"
        "  - Factor Pendiente:  f_s = exp( c_s * (dh / dist) )\n"
        "  - Factor Viento:     f_w = exp( c_w * (V_vec · D_vec) )\n\n"
        "• Naturaleza Estocastica (Metodo Monte Carlo):\n"
        "  La probabilidad de salto estocastico reproduce turbulencias,\n"
        "  chispas y piroconveccion real en bosques tropicales."
    )
    p3_ax.text(0.06, 0.46, text_p3, fontsize=10.2, color=C_TEXT, linespacing=1.30, va="center", transform=p3_ax.transAxes)

    # -------------------------------------------------------------
    # 5. PANEL DERECHO CENTRAL: MAPA VISUAL DE SIMULACION EN GUATEMALA
    # -------------------------------------------------------------
    p4_ax = fig.add_axes([0.52, 0.36, 0.43, 0.27], facecolor=C_CARD)
    p4_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p4_ax.transAxes))
    p4_ax.axis("off")
    p4_ax.text(0.06, 0.92, "4. MAPA DE SALIDA: PROPAGACION EN GUATEMALA", fontsize=13, fontweight="bold", color=C_ACCENT_BLUE, transform=p4_ax.transAxes)

    # Sub-eje embebido para la gráfica de la simulación
    map_sub = fig.add_axes([0.54, 0.375, 0.39, 0.21])
    
    # Generar simulación rápida representativa
    s = 200
    terrain = TerrainGrid.create_synthetic_guatemala(rows=s, cols=s)
    wind = WindField.uniform(shape=(s, s), speed_ms=12.0, direction_degrees=45.0)
    sim = FireSimulation(terrain, wind)
    sim.ignite(int(s * 0.75), int(s * 0.5))
    sim.ignite(int(s * 0.35), int(s * 0.45))
    sim.run(steps=15)
    
    # Renderizar mapa
    map_sub.imshow(sim.terrain.elevation, cmap="terrain", origin="upper")
    
    # Capa de fuego
    mask = np.zeros((*sim.state.shape, 4))
    mask[sim.state == STATE_BURNED] = [0.1, 0.1, 0.1, 0.75]
    mask[sim.state == STATE_BURNING] = [1.0, 0.2, 0.0, 0.95]
    map_sub.imshow(mask, origin="upper")
    
    # Vectores de viento reducidos
    step_skip = 15
    y_idxs, x_idxs = np.mgrid[0:s:step_skip, 0:s:step_skip]
    map_sub.quiver(x_idxs, y_idxs, wind.u[0:s:step_skip, 0:s:step_skip], -wind.v[0:s:step_skip, 0:s:step_skip],
                   color="#38BDF8", alpha=0.6, scale=120, width=0.003)
    map_sub.set_xticks([])
    map_sub.set_yticks([])
    map_sub.set_title("Orografia, Viento Alisio (45°) y Frentes Activos", fontsize=9, color=C_TEXT, pad=4)

    # -------------------------------------------------------------
    # 6. PANEL IZQUIERDO INFERIOR: METRICAS Y BENCHMARK CUANTITATIVO
    # -------------------------------------------------------------
    p5_ax = fig.add_axes([0.05, 0.06, 0.43, 0.28], facecolor=C_CARD)
    p5_ax.axis("off")
    p5_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p5_ax.transAxes))
    
    p5_ax.text(0.06, 0.92, "5. BENCHMARK CUANTITATIVO (CPU BASELINE)", fontsize=13, fontweight="bold", color=C_ACCENT_ORANGE, transform=p5_ax.transAxes)

    # Sub-eje embebido para el gráfico de barras/lineas
    chart_sub = fig.add_axes([0.08, 0.09, 0.37, 0.18], facecolor="#131B2E")
    
    # Datos medidos reales
    cell_labels = ["200x200\n(40k)", "500x500\n(250k)", "1000x1000\n(1M celdas)"]
    times = [0.0367, 0.1775, 0.6486]
    
    bars = chart_sub.bar(cell_labels, times, color=["#38BDF8", "#F59E0B", "#EF4444"], width=0.55, edgecolor=C_BORDER)
    chart_sub.set_ylabel("Tiempo por corrida (s)", color=C_MUTED, fontsize=9)
    chart_sub.tick_params(colors=C_MUTED, labelsize=8.5)
    chart_sub.grid(axis="y", linestyle="--", alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        chart_sub.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f} s",
                       ha="center", va="bottom", color=C_TEXT, fontsize=8.5, fontweight="bold")
        
    chart_sub.set_title("Crecimiento de Tiempo en CPU Monohilo", color=C_TEXT, fontsize=10, pad=6)

    # -------------------------------------------------------------
    # 7. PANEL DERECHO INFERIOR: HOJA DE RUTA Y SUPERCÓMPUTO KABRÉ
    # -------------------------------------------------------------
    p6_ax = fig.add_axes([0.52, 0.06, 0.43, 0.28], facecolor=C_CARD)
    p6_ax.axis("off")
    p6_ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                          facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.2, transform=p6_ax.transAxes))
    
    p6_ax.text(0.06, 0.92, "6. HOJA DE RUTA EN KABRE (CeNAT)", fontsize=13, fontweight="bold", color=C_ACCENT_GREEN, transform=p6_ax.transAxes)
    
    text_p6 = (
        "• Bloque Matutino: Paralelismo y Optimizacion en CPU:\n"
        "  - Compilacion Just-in-Time con Numba (@jit(parallel=True)).\n"
        "  - Distribucion de filas entre nucleos del nodo con prange.\n"
        "  - Medicion de curvas de escalamiento fuerte y debil (Ley de Amdahl).\n\n"
        "• Bloque Vespertino: Aceleracion Masiva en GPU:\n"
        "  - Mapeo de arreglos matriciales a VRAM con CuPy (cp.ndarray).\n"
        "  - Kernel CUDA 2D Stencil donde 1 hilo de GPU procesa 1 celda.\n"
        "  - Reduccion del tiempo de transferencia Host-a-Device.\n\n"
        "• Proyeccion a Escala Nacional (Guatemala Completa):\n"
        "  - Resolucion: 20 millones de celdas (malla de 4,500 x 4,500).\n"
        "  - Laptop convencional: Inviable (> 24 horas para 500 corridas).\n"
        "  - GPU en Kabre: < 30 segundos por lote Monte Carlo."
    )
    p6_ax.text(0.06, 0.46, text_p6, fontsize=10.0, color=C_TEXT, linespacing=1.35, va="center", transform=p6_ax.transAxes)

    # Guardar poster
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[OK] Poster academico generado exitosamente en: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_academic_poster("YAXCHE_POSTER.png")
