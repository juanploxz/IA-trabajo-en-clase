# IA: ejercicios de aprendizaje por refuerzo y clasificación

Repositorio académico en Python con cinco ejercicios independientes de
inteligencia artificial. Incluye planificación mediante MDP, aprendizaje por
refuerzo, reconocimiento de dígitos y clasificación supervisada.

Cada ejercicio tiene un ejecutable propio, documentación, pruebas automáticas
y resultados reproducibles. La explicación conjunta y ampliada está en
[DOCUMENTACION_COMPLETA.md](DOCUMENTACION_COMPLETA.md).

## Ejercicios

| # | Ejercicio | Método | Ejecutable | Documentación |
|---:|---|---|---|---|
| 1 | GridWorld 15×15 | Value Iteration | `main.py` | [Guía](VALUE_ITERATION_README.md) |
| 2 | Reconocimiento de MNIST | REINFORCE como bandido contextual | `mnist_rl.py` | [Guía](MNIST_RL_README.md) |
| 3 | GridWorld 15×15 | Q-Learning y comparación | `q_learning_grid.py` | [Guía](Q_LEARNING_README.md) |
| 4 | Clasificación de Iris | LDA, K-NN(3) y árbol | `iris_classification.py` | [Guía](IRIS_CLASSIFICATION_README.md) |
| 5 | Particiones de Iris | Comparación 60/40, 70/30 y 80/20 | `iris_split_comparison.py` | [Guía](IRIS_SPLIT_COMPARISON_README.md) |

## Resumen de cada ejercicio

### 1. Value Iteration

Resuelve un GridWorld determinista de 15×15 con cinco obstáculos. El algoritmo
conoce el modelo del entorno, actualiza los valores con Bellman y extrae una
política óptima. Con la configuración predeterminada converge en 29 iteraciones
y encuentra una ruta de 28 pasos con recompensa acumulada 73.

### 2. MNIST con REINFORCE

Formula el reconocimiento de dígitos como un bandido contextual: la imagen es
el estado, elegir un número es la acción y la etiqueta determina la recompensa.
Permite descargar MNIST, entrenar la política, consultar ejemplos por número,
reconocer imágenes externas y utilizar una interfaz gráfica.

### 3. Q-Learning

Aprende por experiencia sobre el mismo GridWorld del primer ejercicio. La
comparación muestra la diferencia entre planificación con modelo y aprendizaje
sin modelo. Después de 5 000 episodios, Q-Learning obtiene una ruta con la misma
longitud y recompensa que Value Iteration.

### 4. Clasificación de Iris

Aplica una partición estratificada 70/30 y compara LDA, K-NN con tres vecinos y
árbol de decisión. Genera accuracy, precisión macro, recall macro, F1 macro,
matrices de confusión y fronteras de decisión bidimensionales.

### 5. Comparación de particiones Iris

Repite los tres clasificadores con divisiones 60/40, 70/30 y 80/20. El ejercicio
muestra el compromiso entre disponer de más datos para aprender y conservar un
conjunto de prueba suficientemente grande para evaluar de forma estable.

## Instalación

Requiere Python 3.10 o superior.

### Windows PowerShell

```powershell
git clone https://github.com/juanploxz/IA-trabajo-en-clase.git
cd IA-trabajo-en-clase
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux o macOS

```sh
git clone https://github.com/juanploxz/IA-trabajo-en-clase.git
cd IA-trabajo-en-clase
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Ejecución rápida

Los comandos siguientes usan PowerShell. En Linux o macOS sustituya
`.\.venv\Scripts\python.exe` por `./.venv/bin/python`.

```powershell
# Value Iteration
.\.venv\Scripts\python.exe main.py

# Preparar, entrenar y abrir MNIST
.\.venv\Scripts\python.exe mnist_rl.py preparar
.\.venv\Scripts\python.exe mnist_rl.py entrenar --epochs 10
.\.venv\Scripts\python.exe mnist_rl.py gui

# Probar el reconocimiento con la imagen incluida
.\.venv\Scripts\python.exe mnist_rl.py predecir examples\numero_7.png

# Q-Learning y comparación
.\.venv\Scripts\python.exe q_learning_grid.py --no-gui

# Clasificación Iris 70/30
.\.venv\Scripts\python.exe iris_classification.py

# Comparación Iris 60/40, 70/30 y 80/20
.\.venv\Scripts\python.exe iris_split_comparison.py
```

Los comandos que aceptan `--no-gui` guardan resultados sin abrir ventanas.
Use `--help` en cada ejecutable para consultar sus opciones.

## Resultados destacados

| Experimento | Resultado verificado |
|---|---:|
| Pruebas automáticas | 35/35 correctas |
| Value Iteration | 28 pasos; recompensa 73 |
| MNIST REINFORCE | 92,58 % de exactitud |
| Q-Learning | 28 pasos; recompensa 73 |
| Mejor F1 de Iris 70/30 | LDA: 97,78 % |
| Mejor F1 en la comparación de particiones | LDA 80/20: 100 % sobre 30 pruebas |

El 100 % de LDA con 80/20 corresponde a una única partición pequeña y no
demuestra superioridad general. La documentación explica esta limitación y
propone validación cruzada para comparaciones futuras.

## Resultados visuales

### Value Iteration

![GridWorld resuelto con Value Iteration](outputs/resultado_final.png)

### Q-Learning frente a Value Iteration

![Comparación de Q-Learning y Value Iteration](outputs/comparacion_q_learning.png)

### Clasificadores de Iris

![Métricas de los clasificadores de Iris](outputs/iris_metricas.png)

### Efecto de la partición de Iris

![Comparación 60/40, 70/30 y 80/20](outputs/iris_particiones_metricas.png)

Los PNG y CSV de referencia están en `outputs/`. Las nuevas ejecuciones escriben
en `output/`, carpeta ignorada por Git salvo su archivo `.gitkeep`.

## Estructura

```text
.
├── main.py
├── mnist_rl.py
├── q_learning_grid.py
├── iris_classification.py
├── iris_split_comparison.py
├── gridworld/                 # Entorno y algoritmos tabulares
├── iris_classifier/           # Modelos, métricas y visualizaciones de Iris
├── tests/                     # 35 pruebas automáticas
├── examples/                  # Imagen de ejemplo para MNIST
├── outputs/                   # Resultados de referencia visibles en GitHub
├── data/mnist/.gitkeep        # Datos descargados localmente
├── models/.gitkeep            # Modelos entrenados localmente
└── requirements.txt
```

Los datos de MNIST, modelos entrenados, entorno virtual, cachés y resultados
temporales no se versionan. Se recrean mediante los comandos documentados.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

La batería cubre transiciones y Bellman, entrenamiento y persistencia de
REINFORCE, actualización TD de Q-Learning, particiones estratificadas,
clasificadores, métricas, reproducibilidad y generación de PNG/CSV.
