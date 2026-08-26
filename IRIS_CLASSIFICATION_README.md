# Clasificación del conjunto Iris

Este ejercicio es independiente de GridWorld, MNIST y Q-Learning. Compara tres
clasificadores supervisados sobre la base Iris:

- Linear Discriminant Analysis (**LDA**).
- K-Nearest Neighbors con `k = 3` (**K-NN(3)**).
- Árbol de decisión.

La partición es estratificada: 70 % para entrenamiento y 30 % para prueba.

## Ejecutar

```powershell
.\.venv\Scripts\python.exe iris_classification.py --no-gui
```

Para mostrar las figuras al terminar:

```powershell
.\.venv\Scripts\python.exe iris_classification.py
```

## Metodología

Iris contiene 150 flores, 50 por especie, y cuatro atributos medidos en
centímetros. La semilla predeterminada es `42`, y `stratify` garantiza:

| Conjunto | Total | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|---:|
| Entrenamiento | 105 | 35 | 35 | 35 |
| Prueba | 45 | 15 | 15 | 15 |

Las métricas principales se calculan con los cuatro atributos. Las fronteras de
decisión necesitan un plano, por lo que se entrenan clones separados usando
longitud y ancho del pétalo. Esta diferencia se indica dentro de la figura para
no confundir una visualización 2D con la evaluación completa 4D.

## Modelos

### LDA

Busca combinaciones lineales que separen las clases considerando medias y
covarianzas. Produce fronteras lineales.

### K-NN(3)

Clasifica según las tres muestras de entrenamiento más cercanas. Usa
`StandardScaler` dentro de un `Pipeline` para evitar que una variable domine la
distancia por su escala. Sus fronteras son locales y no lineales.

### Árbol de decisión

Divide el espacio mediante reglas sobre atributos. No necesita escalado y puede
crear fronteras rectangulares no lineales. Se fija `random_state` para asegurar
reproducibilidad.

## Métricas

- **Accuracy:** proporción total de predicciones correctas.
- **Precisión macro:** promedio de precisión calculado por especie.
- **Recall macro:** promedio de sensibilidad por especie.
- **F1 macro:** media armónica macro de precisión y recall.
- **Matriz de confusión:** cruza clases reales y predichas.

El promedio macro concede el mismo peso a cada clase. Iris está balanceado, pero
esta práctica mantiene una evaluación clara por especie.

## Argumentos

| Argumento | Descripción | Predeterminado |
|---|---|---:|
| `--test-size X` | Proporción reservada para prueba | `0.30` |
| `--seed N` | Semilla de la partición y del árbol | `42` |
| `--neighbors N` | Número de vecinos de K-NN | `3` |
| `--tree-max-depth N` | Profundidad máxima del árbol | Sin límite |
| `--features X Y` | Índices para las fronteras | `2 3` |
| `--output-dir RUTA` | Carpeta de resultados | `output` |
| `--no-gui` | Guarda sin abrir ventanas | Desactivado |

Índices de atributos:

```text
0 = longitud del sépalo
1 = ancho del sépalo
2 = longitud del pétalo
3 = ancho del pétalo
```

## Archivos generados

```text
output/iris_fronteras_decision.png
output/iris_metricas.png
output/iris_metricas.csv
output/iris_predicciones.csv
```

El CSV de predicciones conserva el índice original de Iris, sus cuatro
atributos, la clase real y las predicciones de los tres modelos.

## Resultados verificados

Ejecución realizada con la configuración predeterminada: partición 70/30,
semilla `42`, K-NN con `k=3` y árbol sin límite de profundidad.

| Clasificador | Accuracy | Precisión macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| LDA | 97,78 % | 97,92 % | 97,78 % | 97,78 % |
| K-NN (k=3) | 91,11 % | 92,98 % | 91,11 % | 90,95 % |
| Árbol de decisión | 93,33 % | 94,44 % | 93,33 % | 93,27 % |

Las matrices de confusión fueron:

```text
LDA                  K-NN (k=3)          Árbol de decisión
[[15, 0,  0],        [[15, 0,  0],       [[15, 0,  0],
 [ 0, 15, 0],         [ 0, 15, 0],        [ 0, 12, 3],
 [ 0, 1, 14]]         [ 0, 4, 11]]        [ 0, 0, 15]]
```

LDA produjo el F1 macro más alto en esta partición y falló solamente una flor
Virginica, clasificada como Versicolor. K-NN confundió cuatro Virginica con
Versicolor, mientras el árbol confundió tres Versicolor con Virginica. El
resultado depende de la partición; para comparar estabilidad entre modelos se
recomienda validación cruzada o varias semillas.

En la visualización bidimensional, las exactitudes fueron 91,11 % para LDA y
93,33 % para K-NN y el árbol. Estas cifras 2D no sustituyen las métricas de la
tabla, calculadas con los cuatro atributos.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_iris_classifier -v
```

Las seis pruebas de este ejercicio verifican la partición 70/30,
estratificación, K-NN con `k=3`, métricas, reproducibilidad, modelos
bidimensionales y generación real de PNG y CSV mediante la CLI. Junto con los
ejercicios anteriores, la suite completa contiene 35 pruebas.

## Cómo exponer el ejercicio

1. Presente las 150 muestras, tres especies y cuatro atributos.
2. Explique por qué se usa una partición estratificada.
3. Compare el supuesto lineal de LDA, la vecindad de K-NN y las reglas del
   árbol.
4. Aclare por qué la evaluación usa cuatro variables y la gráfica solo dos.
5. Interprete accuracy, métricas macro y matrices de confusión.
6. Compare la forma geométrica de las tres fronteras.

## Mejoras futuras

- Añadir validación cruzada estratificada.
- Ajustar hiperparámetros con GridSearchCV.
- Comparar normalización y selección de atributos.
- Añadir ROC multiclase y curvas de aprendizaje.
- Exportar los modelos entrenados para inferencia posterior.
