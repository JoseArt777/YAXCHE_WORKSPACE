# YAXCHÉ
**Simulador de Incendios Forestales en Guatemala con Supercomputación (HPC)**

[![HPC: Kabré Cluster](https://img.shields.io/badge/HPC-Kabré%20(CeNAT)-blue.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Acceleration: Numba & CUDA](https://img.shields.io/badge/Acceleration-Numba%20%7C%20CUDA-orange.svg)](#)
[![Data: NASA Earth Science](https://img.shields.io/badge/Data-NASA%20Earth-red.svg)](#)

---

## 1. Qué es YAXCHÉ y por qué existe

En la cosmovisión maya, la **Yaxché (Ceiba Pentandra)** es el Árbol de la Vida y el árbol nacional de Guatemala, símbolo de conexión entre el cielo y la tierra.

El proyecto **YAXCHÉ** toma esa idea y la convierte en una herramienta práctica: **toma datos reales de satélites (cielo), los procesa en el supercomputador Kabré y proyecta hacia dónde va a avanzar un incendio en los bosques de Guatemala (tierra)** antes de que sea demasiado tarde.

### El problema que resuelve
Cada época seca (febrero a junio), Guatemala pierde miles de hectáreas de bosque en lugares críticos como la **Reserva de la Biosfera Maya (Petén)**, la **Sierra de las Minas** y la **Cadena Volcánica**.

Hoy en día existen dos limitaciones principales:
1. **Los satélites avisan tarde:** reportan dónde *ya hay* fuego, pero no muestran hacia dónde se moverá en las próximas **24, 48 o 72 horas**.
2. **Simular todo un país con detalle es demasiado lento para una computadora normal:** si divides Guatemala en cuadros pequeños de ~250 metros, obtienes **4 millones de cuadros**. Calcular qué pasa en esos 4 millones de cuadros cientos de veces para probar distintos escenarios de viento puede tomar demasiado tiempo cuando hay una emergencia activa.

---

## 2. Cómo funciona desde cero (sin tecnicismos)

Piensa en YAXCHÉ como un mapa digital de Guatemala dividido en una cuadrícula gigante de **4,000,000 de cuadritos** (`2000 x 2000`), donde cada cuadrito mide **249 metros por lado**.

Para predecir cómo avanza el fuego, el sistema mezcla **tres ingredientes reales**:

1. **El relieve del terreno (qué tan inclinado está y dónde hay agua):**
   - Descargado de mapas topográficos satelitales de la NASA/AWS.
   - El fuego sube mucho más rápido cuesta arriba que cuesta abajo, y se detiene por completo cuando llega a lagos o ríos grandes (como el Lago de Atitlán o el Lago de Izabal).
2. **El viento real (velocidad y dirección):**
   - Consultado desde **Open-Meteo** (datos históricos o en vivo de hoy).
   - El viento empuja las llamas: si sopla del Este, el fuego se estira hacia el Oeste; si cambia hacia el Sureste, la pluma de fuego gira hacia el Sureste.
3. **Los focos de calor iniciales (dónde empezó el fuego):**
   - Tomados de los satélites **NASA FIRMS (VIIRS)**.
   - En épocas activas (como el **18 de abril de 2024**), el satélite detecta decenas de incendios reales en Petén, Sierra de las Minas y el Altiplano, que sirven como punto de partida.

### Qué ocurre dentro de la simulación
1. **Paso a paso por hora:** en cada hora simulada (por ejemplo, 48 horas hacia adelante), cada cuadrito que está ardiendo evalúa a sus 8 vecinos y calcula la probabilidad de contagiarles fuego según la vegetación, la pendiente del terreno y la dirección del viento.
2. **200 futuros posibles (Método Monte Carlo):** como el clima nunca es 100% predecible, YAXCHÉ no corre la simulación una sola vez. Corre **200 simulaciones distintas**, cambiando ligeramente la fuerza y dirección del viento en cada una.
3. **Mapa final de riesgo:** al juntar las 200 simulaciones, cada lugar del mapa recibe un porcentaje de probabilidad (de 0% a 100%): si en 180 de 200 simulaciones el fuego llegó a una comunidad o reserva, ese punto aparece en rojo oscuro (**90% de riesgo**).

---

## 3. Cómo se aprovecha el HPC (Computación de Alto Rendimiento)

Aquí es donde entra el supercomputador **Kabré (CeNAT, Costa Rica)**. Para entender qué hace el HPC, imagina que tienes que revisar los **4 millones de cuadros** del mapa:

1. **Sin aceleración (1 trabajador en fila):**
   - Un programa convencional recorre los 4 millones de cuadros **uno detrás de otro**, y repite eso 48 veces (48 horas).
2. **CPU Paralelo con Numba (`4` trabajadores):**
   - Compilamos el código Python a instrucciones nativas del procesador y repartimos las filas del mapa entre varios núcleos del procesador al mismo tiempo.
3. **GPU Masivamente Paralela con CUDA (`NVIDIA L40S`):**
   - En lugar de unos pocos núcleos trabajando por turnos, la tarjeta gráfica **NVIDIA L40S** asigna **miles de mini-procesadores trabajando al mismo instante** (un hilo de GPU por cada cuadro del mapa) y mantiene toda la simulación dentro de su memoria de alta velocidad (VRAM).
   - Además, ordenamos los datos en memoria para que la GPU lea bloques contiguos de una sola vez (**acceso coalescente**) en lugar de saltar posiciones (**acceso strided**). Solo ese ordenamiento redujo el tiempo de cálculo en **3.86 veces**.

### Resultados reales medidos en Kabré (`nukwa-06.cnca`, Job `634819`)
Malla real de Guatemala **`2000 x 2000`** (`4,000,000` celdas de `249 m`, `48` horas simuladas):

| Modo de ejecución | Tiempo (mediana) | Velocidad vs CPU 1 hilo |
| --- | ---: | ---: |
| **CPU Numba (1 hilo)** | `1,126.98 ms` | `1.0x` (referencia) |
| **CPU Numba (2 hilos)** | `636.73 ms` | `1.77x` más rápido |
| **CPU Numba (4 hilos)** | `435.76 ms` | `2.59x` más rápido |
| **GPU L40S (incluyendo copia de memoria)** | `19.974 ms` | **`56.4x` más rápido** |
| **GPU L40S (solo cálculo, memoria coalescente)** | `1.103 ms` | **`1,021.7x` más rápido** |

- **Exactitud verificada:** tanto la versión en CPU como las variantes en GPU (coalescente y strided) produjeron exactamente el mismo número de celdas afectadas (**`7,001` celdas**), comprobando que acelerar el cálculo no altera el resultado científico.
- **Ensamble Monte Carlo completo:** correr **200 simulaciones** sobre toda Guatemala a 249 m tomó **`11.11 segundos`** en la GPU (`55.5 ms` por simulación).

---

## 4. Qué muestran los resultados visuales

- **Mapa probabilístico de riesgo ([results/riesgo_montecarlo.png](results/riesgo_montecarlo.png)):**
  - Muestra las **200 réplicas** sobre Guatemala (`2000 x 2000`, `dx = 249 m`) a partir de `18` focos reales del `2024-04-18`, proyectando **`41,696 ha`** de riesgo esperado en 48 horas, junto con dos paneles de acercamiento táctico para **Petén (RBM)** y el **Altiplano / Sierra de las Minas**.
- **Prueba de sensibilidad con viento en vivo ([results/comparativa_viento_vivo.png](results/comparativa_viento_vivo.png)):**
  - Compara los mismos focos usando el viento seco de abril 2024 (`3.6 m/s` desde el Este, `43,334 ha`) frente al viento en vivo consultado en Open-Meteo (`1.8 m/s` desde el Noroeste, `27,581 ha`), demostrando que el fuego cambia de dirección y extensión en tiempo real según el clima.
- **Póster científico oficial ([YAXCHE_POSTER_OFICIAL.png](YAXCHE_POSTER_OFICIAL.png)):**
  - Resume el problema nacional, la física del modelo, las gráficas de rendimiento en Kabré (`450x450`, `2000x2000` y `4500x4500`) y el impacto operativo.

---

## 5. Para qué servirá este proyecto en el futuro

1. **Respuesta rápida durante incendios reales (CONRED, CONAP, INAB):** en lugar de esperar reportes tardíos, los equipos en tierra pueden obtener en ~11 segundos un mapa de hacia dónde avanzará el fuego en las siguientes 48 horas y priorizar la protección de comunidades o reservas naturales.
2. **Prevención antes de la temporada seca:** permite probar miles de escenarios de viento antes de marzo para decidir dónde construir brechas cortafuego y dónde ubicar brigadas preventivamente.
3. **Base lista para crecer:** la misma arquitectura en GPU puede conectarse en el futuro con pronósticos horarios de viento en vivo, mapas diarios de humedad de vegetación o modelos de dispersión de humo.

---

## 6. Estructura del proyecto

```text
YAXCHE/
├── data/
│   ├── raw/
│   │   └── nasa_firms_viirs_guatemala_2024.csv   # Focos reales NASA FIRMS 2024
│   └── processed/
│       └── guatemala_real.npz                    # Malla real de Guatemala (450x450)
├── jobs/
│   ├── yaxche_cpu.sbatch                         # Job Slurm para CPU en Kabré
│   └── yaxche_gpu.sbatch                         # Job Slurm para GPU en Kabré
├── results/
│   ├── benchmarks_kabre.jsonl                    # 29 mediciones reales en Kabré
│   ├── riesgo_montecarlo.png                     # Mapa Monte Carlo + zooms tácticos
│   └── comparativa_viento_vivo.png               # Comparación viento histórico vs en vivo
├── scripts/
│   ├── build_real_dataset.py                     # Descarga DEM, viento y focos reales
│   └── benchmark_kabre.py                        # Suite de benchmarks CPU/GPU y Monte Carlo
├── src/
│   ├── terrain.py                                # Frontera de Guatemala, relieve y lagos
│   ├── wind.py                                   # Campo de viento 2D
│   ├── simulation.py                             # Simulador base (referencia)
│   ├── simulation_numba.py                       # Simulador paralelo en CPU (Numba)
│   ├── simulation_gpu.py                         # Simulador paralelo en GPU (CuPy / CUDA)
│   └── visualization.py                          # Gráficas cartográficas de reporte
├── tests/
│   └── test_simulation.py                        # Pruebas de validación
├── generate_poster.py                            # Generador del póster oficial
├── YAXCHE_POSTER_OFICIAL.png                     # Póster final en alta resolución
└── run_demo.py                                   # Demostración rápida local
```

---

## 7. Cómo ejecutarlo paso a paso

### En una computadora normal (local)
```bash
git clone https://github.com/JoseArt777/YAXCHE_WORKSPACE.git
cd YAXCHE_WORKSPACE/YAXCHE
pip install -r requirements.txt

# 1. Correr simulación rápida de 48 horas
python3 run_demo.py

# 2. (Opcional) Generar dataset real de alta resolución (2000x2000, ~249 m/celda)
python3 scripts/build_real_dataset.py --size 2000 --zoom 10 --out data/processed/guatemala_real_2000.npz
```

### En el Clúster Kabré (GPU NVIDIA L40S)
```bash
module load mamba/Python4HPC
python scripts/benchmark_kabre.py --mode gpu --npz data/processed/guatemala_real_2000.npz --steps 48 --repeat 10 --mc-runs 200
python scripts/benchmark_kabre.py --mode cpu --npz data/processed/guatemala_real_2000.npz --steps 48 --repeat 3
```

---

## 8. Reconocimientos
Proyecto desarrollado en el marco del **I Taller de Computación de Alto Rendimiento (HPC) en CPU y GPU con Python**, en colaboración con el **Centro Nacional de Alta Tecnología (CeNAT / Clúster Kabré, Costa Rica)** y la **Facultad de Ingeniería (ECYS) de la Universidad de San Carlos de Guatemala (USAC)**.
