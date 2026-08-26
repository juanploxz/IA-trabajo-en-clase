# Subajuste y sobreajuste con Wine y Breast Cancer

Este ejercicio independiente compara configuraciones de **K-NN** y **árbol de
decisión** que pueden producir subajuste, un ajuste razonable o sobreajuste. Se
ejecuta sobre dos conjuntos incluidos en scikit-learn:

- **Wine:** 178 vinos, 13 atributos químicos y 3 cultivares.
- **Breast Cancer Wisconsin:** 569 muestras, 30 atributos y 2 clases
  (maligno/benigno).

Los datos vienen con la biblioteca, por lo que este ejercicio no descarga
archivos ni necesita conexión a internet.

## Ejecutar

Generar los resultados y mostrar las gráficas:

```powershell
.\.venv\Scripts\python.exe classification_fit_comparison.py
```

Generar PNG y CSV sin abrir ventanas:

```powershell
.\.venv\Scripts\python.exe classification_fit_comparison.py --no-gui
```

## Diseño experimental

Se estudian las particiones estratificadas siguientes con semilla `42`:

| Dataset | Partición | Entrenamiento | Prueba |
|---|---:|---:|---:|
| Wine | 50/50 | 89 | 89 |
| Wine | 40/60 | 71 | 107 |
| Breast Cancer | 50/50 | 284 | 285 |
| Breast Cancer | 40/60 | 227 | 342 |

`50/50` y `40/60` son particiones **holdout**, no modalidades de validación
cruzada. Después de reservar la prueba, se aplica `StratifiedKFold` de cinco
pliegues únicamente sobre el entrenamiento. La prueba no participa en el
ajuste, la validación cruzada ni el diagnóstico de complejidad.

La estratificación conserva aproximadamente la proporción de clases en cada
subconjunto. Todos los modelos reciben exactamente las mismas observaciones en
cada combinación dataset/partición.

## Configuraciones comparadas

| Modelo | Complejidad | Comportamiento que se desea examinar |
|---|---|---|
| K-NN `k=1` | Muy flexible | Posible sobreajuste |
| K-NN `k=7` | Intermedia | Configuración solicitada |
| K-NN `k=51` | Muy suavizada | Posible subajuste |
| Árbol profundidad 1 | Muy simple | Posible subajuste |
| Árbol profundidad 5 | Intermedia/alta | Compromiso o sobreajuste |
| Árbol sin límite | Muy flexible | Posible sobreajuste |

K-NN se ejecuta dentro de un `Pipeline` con `StandardScaler`. Esto es necesario
porque el algoritmo usa distancias y los atributos de ambos datasets tienen
escalas diferentes. El escalador se ajusta de nuevo dentro de cada pliegue de
CV, evitando fuga de información.

### Aclaración sobre `k=7`

En K-NN, aumentar `k` **no aumenta la complejidad del modelo**. Ocurre lo
contrario: `k=1` genera una frontera muy flexible; al aumentar `k`, más vecinos
participan en la votación y la frontera se suaviza. Un valor demasiado grande,
como `k=51` en estas particiones, puede ignorar patrones locales y subajustar.

En un árbol sí aumenta la flexibilidad cuando aumenta la profundidad. Un árbol
de profundidad 1 puede no capturar la estructura, mientras que un árbol sin
límite puede memorizar el entrenamiento.

## Métricas

Para cada modelo se calculan:

- Accuracy de entrenamiento.
- Accuracy, balanced accuracy, precisión macro, recall macro y F1 macro de
  prueba.
- Accuracy de entrenamiento interno y validación en cada pliegue.
- F1 macro de validación cruzada.
- Media y desviación estándar de CV.
- Brecha de prueba: `accuracy_train - accuracy_test`.
- Brecha de CV: `media_train_CV - media_validación_CV`.
- Matriz de confusión, conservada en memoria para validación interna.

Una brecha grande entre entrenamiento y validación/prueba es una señal de
sobreajuste. Si entrenamiento y validación quedan bajos frente a las mejores
configuraciones del mismo experimento, hay una señal de subajuste.

El diagnóstico textual del CSV es una **heurística descriptiva**, no una prueba
estadística. Marca subajuste cuando la validación queda al menos 3 puntos
porcentuales debajo de la mejor y la configuración es deliberadamente simple;
marca sobreajuste cuando la brecha de CV alcanza 5 puntos o la brecha holdout
alcanza 8 puntos, salvo que antes exista una señal clara de subajuste.

## Resultados verificados

### Wine

| Partición | Modelo | Train | Test | CV validación | Diagnóstico |
|---|---|---:|---:|---:|---|
| 50/50 | K-NN `k=1` | 100,0 % | 94,4 % | 98,9 % | Competitivo |
| 50/50 | K-NN `k=7` | 98,9 % | 93,3 % | 98,9 % | Competitivo |
| 50/50 | K-NN `k=51` | 89,9 % | 85,4 % | 64,2 % | Subajuste |
| 50/50 | Árbol profundidad 1 | 67,4 % | 61,8 % | 61,8 % | Subajuste |
| 50/50 | Árbol profundidad 5 | 100,0 % | 89,9 % | 92,1 % | Sobreajuste |
| 50/50 | Árbol sin límite | 100,0 % | 89,9 % | 92,1 % | Sobreajuste |
| 40/60 | K-NN `k=1` | 100,0 % | 95,3 % | 97,2 % | Competitivo |
| 40/60 | K-NN `k=7` | 100,0 % | 90,7 % | 97,2 % | Sobreajuste |
| 40/60 | K-NN `k=51` | 71,8 % | 66,4 % | 40,9 % | Subajuste |
| 40/60 | Árbol profundidad 1 | 69,0 % | 60,7 % | 64,9 % | Subajuste |
| 40/60 | Árbol profundidad 5 | 100,0 % | 88,8 % | 84,7 % | Sobreajuste |
| 40/60 | Árbol sin límite | 100,0 % | 88,8 % | 84,7 % | Sobreajuste |

Wine hace muy visible el subajuste: con solo 71 muestras de entrenamiento,
`k=51` deja que casi toda la base de entrenamiento participe en cada votación y
la accuracy media de CV cae a 40,9 %. El árbol de profundidad 1 tampoco puede
representar adecuadamente las tres clases. Por el otro lado, los árboles
profundos memorizan el 100 % del entrenamiento y generalizan peor.

### Breast Cancer Wisconsin

| Partición | Modelo | Train | Test | CV validación | Diagnóstico |
|---|---|---:|---:|---:|---|
| 50/50 | K-NN `k=1` | 100,0 % | 93,7 % | 94,4 % | Sobreajuste |
| 50/50 | K-NN `k=7` | 97,2 % | 96,1 % | 96,1 % | Competitivo |
| 50/50 | K-NN `k=51` | 95,4 % | 93,0 % | 92,2 % | Subajuste |
| 50/50 | Árbol profundidad 1 | 93,3 % | 90,5 % | 89,4 % | Subajuste |
| 50/50 | Árbol profundidad 5 | 99,6 % | 91,6 % | 93,3 % | Sobreajuste |
| 50/50 | Árbol sin límite | 100,0 % | 91,2 % | 93,0 % | Sobreajuste |
| 40/60 | K-NN `k=1` | 100,0 % | 94,7 % | 95,6 % | Competitivo |
| 40/60 | K-NN `k=7` | 97,4 % | 95,3 % | 96,0 % | Competitivo |
| 40/60 | K-NN `k=51` | 94,7 % | 92,1 % | 92,5 % | Subajuste |
| 40/60 | Árbol profundidad 1 | 94,3 % | 90,1 % | 90,7 % | Subajuste |
| 40/60 | Árbol profundidad 5 | 100,0 % | 91,2 % | 95,6 % | Sobreajuste |
| 40/60 | Árbol sin límite | 100,0 % | 91,2 % | 95,6 % | Sobreajuste |

En Breast Cancer, `k=7` ofrece el mejor compromiso: obtiene 96,1 % de prueba
con 50/50 y 95,3 % con 40/60, sin una brecha grande. Los árboles profundos
llegan casi siempre a 100 % de entrenamiento, pero permanecen entre 91 % y
93 % en prueba/CV, señal de varianza y memorización.

## Qué muestran las gráficas

1. `clasificadores_ajuste_metricas.png`: train, prueba y media de CV con su
   desviación estándar para las 24 combinaciones.
2. `clasificadores_ajuste_knn.png`: curva completa para
   `k=1, 3, 5, 7, 11, 15, 21, 31, 41, 51`; resalta `k=7`.
3. `clasificadores_ajuste_arbol.png`: curva para profundidades
   `1, 2, 3, 5, 8` y sin límite.

Las curvas permiten ver que no se debe escoger un hiperparámetro solo porque
mejora el entrenamiento. La región útil es aquella donde CV es alta y la
separación respecto al entrenamiento permanece controlada.

## Archivos generados

```text
output/clasificadores_ajuste_metricas.png
output/clasificadores_ajuste_knn.png
output/clasificadores_ajuste_arbol.png
output/clasificadores_ajuste_metricas.csv
output/clasificadores_ajuste_cv.csv
output/clasificadores_ajuste_complejidad.csv
output/clasificadores_ajuste_predicciones.csv
```

El CSV de métricas tiene 24 filas; el detalle de CV contiene 120 filas; la
curva de complejidad contiene 64 puntos y las predicciones reúnen 823
evaluaciones holdout.

## Argumentos

| Argumento | Descripción | Predeterminado |
|---|---|---:|
| `--seed N` | Semilla de partición, CV y árboles | `42` |
| `--cv-folds N` | Cantidad de pliegues estratificados | `5` |
| `--reference-k N` | Valor intermedio señalado en la gráfica | `7` |
| `--high-k N` | Valor grande para examinar subajuste | `51` |
| `--train-sizes P...` | Proporciones de entrenamiento | `0.50 0.40` |
| `--output-dir RUTA` | Directorio de resultados | `output` |
| `--no-gui` | Guarda sin abrir ventanas | Desactivado |

Si se reduce mucho el entrenamiento o se usan pocos pliegues, `k=51` puede ser
mayor que el entrenamiento interno de un fold. El programa detecta esa
situación y explica el error en lugar de producir una evaluación inválida.

## Pruebas

Ejecutar solo las ocho pruebas de este ejercicio:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_classification_fit -v
```

Las pruebas comprueban datasets, configuraciones, tamaños, estratificación,
métricas, CV, diagnósticos, curvas, filas CSV y generación de los tres PNG.

## Conclusiones

1. Más complejidad no garantiza mejor generalización.
2. En K-NN la complejidad disminuye al aumentar `k`; en el árbol aumenta con la
   profundidad.
3. Wine sufre especialmente cuando solo se entrena con 40 % y se usa `k=51`.
4. `k=7` resulta un compromiso sólido en Breast Cancer, pero no debe
   seleccionarse únicamente por un holdout.
5. CV reduce la dependencia de una sola división, aunque con pocos datos sus
   pliegues todavía pueden presentar variabilidad.
6. Para selección formal de hiperparámetros conviene usar CV anidada y reservar
   una prueba final completamente independiente.
