# Comparación de modelos de regresión con California Housing

Este ejercicio independiente compara cuatro modelos de regresión para estimar
el valor mediano de viviendas en grupos de bloques censales de California:

- Regresión lineal múltiple.
- Regresión polinomial de grado 2.
- Regresión log-lineal.
- Árbol de decisión para regresión.

Se utiliza **California Housing** porque el antiguo Boston Housing ya no forma
parte de las versiones modernas de `scikit-learn`. La base se descarga
automáticamente la primera vez y después se reutiliza desde
`data/california_housing/`.

## Ejecutar

Generar resultados sin abrir ventanas:

```powershell
.\.venv\Scripts\python.exe california_housing_regression.py --no-gui
```

Mostrar las figuras al finalizar:

```powershell
.\.venv\Scripts\python.exe california_housing_regression.py
```

La primera ejecución necesita conexión a internet. Una vez descargada la base,
puede comprobar que el ejercicio funciona sin descargar usando `--no-download`.

## Base de datos

California Housing contiene 20 640 observaciones y ocho atributos numéricos:

| Índice | Atributo |
|---:|---|
| 0 | Ingreso mediano |
| 1 | Edad mediana de las viviendas |
| 2 | Habitaciones promedio |
| 3 | Dormitorios promedio |
| 4 | Población |
| 5 | Ocupantes promedio |
| 6 | Latitud |
| 7 | Longitud |

El objetivo es el valor mediano de vivienda expresado en cientos de miles de
dólares. Son datos agregados de bloques censales, no precios de propiedades
individuales.

## Diseño experimental

1. Se reserva 80 % para entrenamiento y 20 % para prueba final con semilla 42.
2. La validación cruzada K-Fold de cinco pliegues se aplica solo al conjunto de
   entrenamiento.
3. Cada modelo se ajusta nuevamente con todo el 80 % de entrenamiento.
4. MAE, RMSE y R² se calculan sobre el 20 % que no intervino en los ajustes.

Separar la prueba final de la validación cruzada evita utilizar las mismas
observaciones para seleccionar y reportar el rendimiento final.

## Modelos

### Regresión lineal

Modela el objetivo como una combinación lineal de los ocho atributos. Las
variables se estandarizan antes del ajuste; esto mejora la comparación numérica
sin alterar la capacidad predictiva de una regresión lineal ordinaria.

### Regresión polinomial

Genera términos de grado 2, incluidos cuadrados e interacciones, y ajusta una
regresión lineal sobre ese espacio ampliado. Puede representar curvaturas, pero
también aumenta la complejidad y el riesgo de sobreajuste.

### Regresión log-lineal

Transforma el objetivo antes del ajuste:

```text
log(1 + y) = β₀ + β₁x₁ + ... + β₈x₈
y_estimado = exp(predicción) - 1
```

Todas las métricas se calculan después de regresar a la escala monetaria
original. Esta formulación es distinta de aplicar logaritmos a atributos como
latitud o longitud, que pueden contener valores negativos.

### Árbol de decisión

Divide el espacio mediante reglas. Se usa profundidad máxima 10 y un mínimo de
cinco muestras por hoja para limitar el sobreajuste. No necesita escalado y
puede aprender relaciones no lineales y discontinuas.

## Métricas

- **MAE:** error absoluto medio; menor es mejor.
- **RMSE:** penaliza con más fuerza los errores grandes; menor es mejor.
- **R²:** proporción de variabilidad explicada; mayor es mejor.
- **CV media ± desviación:** estabilidad del modelo entre los cinco pliegues.

MAE y RMSE mantienen la unidad del objetivo: cientos de miles de dólares. Por
ejemplo, un MAE de `0.50` equivale aproximadamente a USD 50 000.

## Líneas de estimación

La figura principal varía el ingreso mediano entre sus percentiles 1 y 99,
mientras mantiene los otros siete atributos en la mediana del conjunto de
entrenamiento. Así se obtiene una línea comparable para cada regresor. No es una
regresión univariada: es una sección del modelo multivariable.

La misma figura incluye cuatro paneles de valor real frente a valor estimado.
La diagonal representa una predicción perfecta y permite observar sesgo,
dispersión y valores difíciles.

## Archivos generados

```text
output/california_regresion_metricas.png
output/california_regresion_estimaciones.png
output/california_regresion_metricas.csv
output/california_regresion_cv.csv
output/california_regresion_predicciones.csv
output/california_regresion_lineas.csv
```

El CSV de métricas contiene el resumen holdout y CV; el de CV conserva cada
pliegue; el de predicciones permite auditar las 4 128 observaciones de prueba;
y el de líneas contiene los puntos exactos usados en la gráfica.

## Resultados verificados

Ejecución con 16 512 muestras de entrenamiento, 4 128 de prueba, semilla `42`
y cinco pliegues sobre el conjunto de entrenamiento:

| Modelo | MAE prueba | RMSE prueba | R² prueba | R² CV media ± desv. |
|---|---:|---:|---:|---:|
| Regresión lineal | 0,5332 | 0,7456 | 0,5758 | 0,6115 ± 0,0138 |
| Regresión polinomial (grado 2) | 0,4670 | 0,6814 | 0,6457 | -1,8847 ± 5,6219 |
| Regresión log-lineal | 0,5362 | 0,9809 | 0,2657 | 0,5212 ± 0,0298 |
| Árbol de decisión | 0,4311 | 0,6380 | 0,6893 | 0,7011 ± 0,0144 |

El árbol obtuvo el menor MAE y RMSE y el mayor R² tanto en prueba como en la
media de validación cruzada. Su MAE equivale aproximadamente a USD 43 100.

La regresión polinomial fue la segunda mejor en la prueba final, pero resultó
inestable en CV: cuatro pliegues tuvieron R² entre 0,49 y 0,68, mientras uno
alcanzó `-11,94`. Ese pliegue elevó el RMSE promedio a 1,3903. Las expansiones
polinomiales sin regularización son sensibles a valores extremos y pueden
extrapolar de forma severa. Este hallazgo muestra por qué no conviene elegir el
modelo usando únicamente una división 80/20.

La regresión lineal fue una referencia estable. La transformación log-lineal no
mejoró este problema: redujo el R² de prueba y produjo el RMSE más alto del
holdout. Una transformación razonable no garantiza una mejor predicción.

## Argumentos

| Argumento | Descripción | Predeterminado |
|---|---|---:|
| `--test-size X` | Proporción de prueba final | `0.20` |
| `--seed N` | Semilla reproducible | `42` |
| `--cv-folds N` | Pliegues de validación cruzada | `5` |
| `--poly-degree N` | Grado del modelo polinomial | `2` |
| `--tree-depth N` | Profundidad máxima del árbol | `10` |
| `--tree-min-leaf N` | Muestras mínimas por hoja | `5` |
| `--line-feature N` | Atributo de la línea estimada | `0` |
| `--data-dir RUTA` | Caché del dataset | `data/california_housing` |
| `--output-dir RUTA` | Resultados | `output` |
| `--no-download` | Impide descargas nuevas | Desactivado |
| `--no-gui` | Guarda sin abrir ventanas | Desactivado |

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_housing_regression -v
```

Las pruebas usan datos sintéticos positivos y no requieren internet. Verifican
los cuatro modelos, partición, reproducibilidad, holdout, validación cruzada,
líneas de estimación, CSV y generación de figuras.

El proyecto completo contiene 42 pruebas automáticas.

## Limitaciones

- Los datos representan bloques censales de 1990 y no el mercado actual.
- El objetivo contiene un límite superior en la base original.
- Una sola partición final no resume toda la incertidumbre del problema.
- Las líneas fijan siete atributos y no muestran todas las interacciones.
- Comparar modelos no sustituye la selección de hiperparámetros en CV anidada.
