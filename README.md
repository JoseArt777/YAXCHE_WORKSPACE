# YAXCHÉ
**Simulador Espacio-Temporal y Acelerado por GPU para la Protección y Propagación de Incendios Forestales en Guatemala**

[![HPC: Kabré Cluster](https://img.shields.io/badge/HPC-Kabré%20(CeNAT)-blue.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Acceleration: Numba & CUDA](https://img.shields.io/badge/Acceleration-Numba%20%7C%20CUDA-orange.svg)](#)
[![Data: NASA Earth Science](https://img.shields.io/badge/Data-NASA%20Earth-red.svg)](#)

---

## 1. Origen e Identidad

En la cosmovisión maya de Mesoamérica, la **Yaxché (la Ceiba Pentandra)** es el *Árbol de la Vida* y el *Axis Mundi* (el eje sagrado del cosmos). Sus ramas superiores se elevan hacia el cielo estrellado, su tronco sostiene la superficie viva y sus raíces descienden a la profundidad de la tierra. La Ceiba es, además, el **Árbol Nacional de Guatemala**, símbolo patrio de longevidad, grandeza y resistencia.

**YAXCHÉ** nace con la misión de emular ese puente cósmico: **captar los datos que descienden desde las alturas del cielo (los sensores satelitales de la NASA), procesarlos a través del clúster de supercómputo Kabré, y aterrizarlos en la masa forestal de Guatemala** para anticipar el avance de frentes de fuego y salvaguardar el pulmón verde de Centroamérica.

---

## 2. Planteamiento del Problema

Cada temporada seca (febrero a junio), Guatemala sufre la destrucción de miles de hectáreas en áreas de altísimo valor ecológico como la **Reserva de la Biosfera Maya (RBM)** en Petén y los bosques montañosos del altiplano central.

### Brecha en las herramientas actuales:
1. **Modelos Reactivos:** Las alertas satelitales institucionales notifican puntos de calor **horas después** de que la ignición ha tenido lugar.
2. **Mapas Estáticos:** Los registros oficiales de recurrencia muestran dónde se quemó en el pasado, pero son incapaces de pronosticar dinámicamente cómo avanzará un incendio en las próximas **24, 48 o 72 horas** bajo condiciones meteorológicas cambiantes.
3. **Cuello de Botella Computacional:** Simular la propagación física a escala nacional (mallas de 100 m de resolución = ~20 millones de celdas) resulta intratable en computadoras convencionales, requiriendo **horas de cálculo** o provocando desbordamientos de memoria (`MemoryError`).

---

## 3. Arquitectura de la Solución

YAXCHÉ implementa un pipeline científico de simulación física acoplado a datos de observación de la Tierra:

```
                    FUENTES DE DATOS ABIERTAS
     +-------------------------------------------------------+
     | - Topografía (DEM): NASA SRTM (30m / 90m)            |
     | - Cobertura & Biomasa: Sentinel-2 / Landsat (NDVI)    |
     | - Detección Térmica: NASA FIRMS (VIIRS 375m & MODIS)  |
     | - Vectores de Viento: NOAA GFS (10m u, v)             |
     +-------------------------------------------------------+
                                |
                                v
     +-------------------------------------------------------+
     |                 MOTOR DE TERRENO Y CLIMA              |
     | - Cálculo tensorial de pendientes y laderas           |
     | - Interpolación de campos de viento y desecación      |
     +-------------------------------------------------------+
                                |
                                v
     +-------------------------------------------------------+
     |            MOTOR DE PROPAGACIÓN CELULAR 2D            |
     | - Autómata celular probabilístico (Vecindad de Moore) |
     | - Modelo semiempírico de avance de frente térmico    |
     +-------------------------------------------------------+
                                |
                                v
       ESCALAMIENTO EN SUPERCÓMPUTO (CLÚSTER KABRÉ - CeNAT)
     +-------------------------------------------------------+
     | [Fase 1] CPU Baseline  : NumPy puro (secuencial)      |
     | [Fase 2] CPU Paralelo  : Numba JIT (multiprocesamiento|
     | [Fase 3] GPU Mass-Par  : CuPy + Kernels CUDA          |
     +-------------------------------------------------------+
```

---

## 4. El Rol del Supercomputador Kabré (CeNAT)

YAXCHÉ ha sido diseñado específicamente para aprovechar las capacidades del clúster **Kabré**:

1. **Paralelismo en Nodo (CPU):**
   * Eliminación de bucles lentos en Python mediante compilación Just-in-Time (`@numba.jit`).
   * Distribución de bloques de celdas geográficas entre los núcleos disponibles (`numba.prange`), evaluando la eficiencia y el factor de aceleración (*Speedup* según la Ley de Amdahl).
2. **Aceleración Masiva en GPU (CUDA):**
   * Mapeo del autómata celular 2D como un **kernel de Stencil espacial**: cada coordenada $(x, y)$ del mapa de Guatemala es evaluada por un hilo de GPU independiente.
   * Optimización del ancho de banda y acceso a memoria compartida (*shared memory*) para calcular la transición de estados en milisegundos.
3. **Escalabilidad Fuerte y Débil:**
   * Evaluación empírica de rendimiento al escalar la resolución del territorio nacional desde 500 m hasta 30 m por celda.

---

## 5. Estructura del Repositorio

```text
YAXCHE/
│
├── data/
│   ├── raw/                 # Datos satelitales crudos (GeoTIFF, NetCDF)
│   └── processed/           # Mallas numéricas preprocesadas (.npy, .npz)
│
├── notebooks/
│   └── 01_baseline_simulacion.ipynb  # Cuaderno de demostración, perfilado y visualización
│
├── src/
│   ├── __init__.py
│   ├── terrain.py           # Modelado de elevación y mapa de combustible
│   ├── wind.py              # Vectores de viento y cizalladura
│   ├── simulation.py        # Motor de propagación celular secuencial y vectorial
│   └── visualization.py     # Renderizado 2D de frentes de fuego e isócronas
│
├── tests/
│   └── test_simulation.py   # Pruebas de integración matemática
│
├── run_demo.py              # Demostración rápida y generación de imágenes
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación del proyecto
```

---

## 6. Inicio Rápido (Entorno Local)

### 1. Clonar el repositorio
```bash
git clone https://github.com/JoseArt777/YAXCHE_WORKSPACE.git
cd YAXCHE_WORKSPACE
```

### 2. Activar entorno virtual e instalar dependencias
```bash
source ../.venv/bin/activate
pip install -r requirements.txt
```

### 3. Ejecutar la simulación base
```bash
python3 run_demo.py
```

O para ver la tabla de benchmark en CPU:
```bash
python3 -m src.simulation
```

O abrir el cuaderno interactivo en Jupyter:
```bash
jupyter notebook notebooks/01_baseline_simulacion.ipynb
```

---

## 7. Licencia y Reconocimientos
Desarrollado como proyecto de investigación aplicada en Computación de Alto Rendimiento (HPC) en colaboración académica con el **I Taller de HPC en CPU y GPU con Python (CeNAT / Kabré)**.
