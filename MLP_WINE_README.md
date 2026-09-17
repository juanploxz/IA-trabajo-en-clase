# Ejercicio 10: comparación de MLP para Wine

Este ejercicio compara la red del [ejercicio 9](ANN_WINE_README.md) con una
red de dos capas ocultas. La anterior ANN **ya es un perceptrón multicapa
(MLP)**: ambas redes son ANN y MLP. La diferencia que se estudia es su
arquitectura, no un cambio de familia de algoritmos.

El programa independiente es [`mlp_wine_compare.py`](mlp_wine_compare.py).

## Ejecutar

Desde la carpeta del proyecto:

```powershell
# Entrenar ambas redes, mostrar resultados y abrir las figuras
.\.venv\Scripts\python.exe mlp_wine_compare.py

# Guardar resultados sin abrir ventanas
.\.venv\Scripts\python.exe mlp_wine_compare.py --no-gui

# Experimentar con otras tasas candidatas
.\.venv\Scripts\python.exe mlp_wine_compare.py --learning-rates 0.001 0.005 0.01 --no-gui
```

Wine viene incluido en scikit-learn y no requiere una descarga adicional.
Si falta el entorno, siga la [instalación general](README.md#instalación).

## Modelos comparados

| Configuración | MLP anterior | MLP nuevo |
|---|---|---|
| Entradas | 13 atributos | 13 atributos |
| Capas ocultas | Una, de 16 neuronas | Dos, de 16 y 8 neuronas |
| Activación oculta | Sigmoide (`logistic`) | Sigmoide en ambas capas |
| Salida | 3 neuronas softmax | 3 neuronas softmax |
| Arquitectura | `13 → 16 → 3` | `13 → 16 → 8 → 3` |
| Pesos | `13×16 + 16×3 = 256` | `13×16 + 16×8 + 8×3 = 360` |
| Sesgos | `16 + 3 = 19` | `16 + 8 + 3 = 27` |
| Total entrenable | 275 parámetros | 387 parámetros |

Todas las neuronas de cada capa se conectan con todas las de la siguiente.
Para la red nueva, después de estandarizar cada atributo con los parámetros
del entrenamiento, la propagación es:

```text
h1 = sigmoid(x_scaled @ W1 + b1)
h2 = sigmoid(h1 @ W2 + b2)
p = softmax(h2 @ W3 + b3)
prediccion = argmax(p)
```

La sigmoide es `1 / (1 + exp(-z))`; softmax produce probabilidades de las
tres clases que suman uno. La segunda capa oculta puede aprender otras
combinaciones de las representaciones previas, pero tener más parámetros
no garantiza mejor generalización.

## Configuración compartida

Para estudiar la arquitectura se mantienen los mismos datos y ajustes:

| Parámetro | Valor |
|---|---|
| Preprocesamiento | `StandardScaler` dentro del pipeline |
| Optimizador | Adam |
| Regularización L2 | `alpha=0.0001` |
| Tamaño de mini-batch | 16 |
| Máximo de épocas | 2 000 |
| Tolerancia | `0.0001` |
| Paciencia | 20 épocas sin mejora suficiente |
| Early stopping con validación interna | Desactivado |
| Semilla | 42 |

Aunque `early_stopping=False`, el entrenamiento puede detenerse antes del
máximo por estancamiento del costo de entrenamiento. Backpropagation obtiene
gradientes respecto a todos los pesos y sesgos, y Adam los actualiza en cada
mini-batch. Son las mismas reglas para las dos arquitecturas. La referencia
del estimador es [MLPClassifier de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

## Split 70/30 y cross validation

Wine tiene 178 muestras, 13 atributos y tres clases. Una única separación
estratificada produce 124 muestras de entrenamiento y 54 de prueba.
Ambas redes usan exactamente esas mismas muestras.

Sobre las 124 de entrenamiento se construyen cinco pliegues mediante
`StratifiedKFold`, con barajado y semilla 42. Se reutilizan los mismos
pliegues para cada combinación de arquitectura y tasa. Cada fold ajusta
su propio escalador con su porción de entrenamiento, evitando fuga desde
la validación.

Cada arquitectura compara las tasas `0.001`, `0.01` y `0.1`. Se selecciona
la de mayor F1 macro medio de validación cruzada; en empate exacto gana la
menor tasa. Después se evalúa el modelo seleccionado de cada arquitectura,
ajustado sobre las 124 muestras, en el test de 54. No se elige la tasa por
el resultado del test ni por el costo de entrenamiento.

Esto compara arquitecturas bajo el mismo procedimiento de selección de
tasa. Si las tasas ganadoras difieren, el resultado incluye esa diferencia
de optimización; para aislar también la tasa se puede repetir con un único
valor mediante `--learning-rates 0.01`.

## ¿Qué es el learning rate?

La tasa de aprendizaje, normalmente escrita `η`, determina la escala del
paso al actualizar parámetros. En descenso de gradiente básico se expresa
como `peso_nuevo = peso_actual - η * gradiente`. Adam usa además estimaciones
acumuladas del gradiente y su cuadrado para adaptar el cambio por parámetro;
esa fórmula simple sirve para entender la escala, no describe todo Adam.

Una tasa pequeña puede necesitar más épocas para reducir el costo. Una tasa
demasiado alta puede producir oscilaciones, saltos o un ajuste peor. La tasa
adecuada depende del modelo y los datos; no se escoge solo porque llegue a
una pérdida pequeña rápidamente.

El argumento que se modifica es `learning_rate_init`, la tasa inicial de
Adam. El parámetro llamado `learning_rate` en `MLPClassifier` controla los
esquemas de SGD y no se usa para esta comparación con Adam. Véase la
[documentación de los parámetros](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

## Costo y métricas

El costo de entrenamiento es entropía cruzada más regularización L2.
Para una muestra cuya clase real es `c`, la entropía cruzada aporta
`-log(p_c)`. No se utiliza error cuadrático medio.

Se reportan accuracy, precisión macro, recall macro, F1 macro, error de
clasificación (`1 - accuracy`) y log-loss. Las métricas macro ponderan cada
clase por igual. El log-loss evalúa las probabilidades finales sin sumar
L2; el costo registrado durante entrenamiento sí incluye regularización.
Por eso costo, log-loss y proporción de errores son cantidades distintas.

Los CSV y las matrices identifican las clases como `0`, `1` y `2`.
En una matriz de confusión, las filas son clases reales y las columnas
clases predichas.

## Resultados reproducibles

Con semilla 42 y los valores predeterminados:

| Métrica | MLP anterior: una capa oculta | MLP nuevo: dos capas ocultas |
|---|---:|---:|
| Tasa seleccionada | 0.001 | 0.001 |
| Accuracy de entrenamiento | 100 % | 100 % |
| Accuracy CV | 98,40 % | 98,40 % |
| F1 macro CV, media ± desv. | 98,34 % ± 2,27 puntos | 98,34 % ± 2,27 puntos |
| Aciertos de prueba | 53/54 | 53/54 |
| Accuracy de prueba | 98,15 % | 98,15 % |
| Precisión macro de prueba | 98,25 % | 98,25 % |
| Recall macro de prueba | 98,41 % | 98,41 % |
| F1 macro de prueba | 98,29 % | 98,29 % |
| Error de clasificación de prueba | 1,85 % | 1,85 % |
| Log-loss de prueba | 0,06670 | 0,06254 |
| Épocas del ajuste final | 323 | 352 |
| Actualizaciones del ajuste final | 2 584 | 2 816 |

Hay ocho mini-batches por época: siete de 16 y uno de 12. Las actualizaciones
de la tabla cuentan únicamente el ajuste final de cada modelo principal,
sin sumar CV, otras tasas o redes auxiliares. La desviación de CV es muestral
(`ddof=1`), no un intervalo de confianza.

Las dos redes predicen exactamente las mismas etiquetas sobre las 54 pruebas,
con la misma matriz de confusión:

```text
              Predicha 0  Predicha 1  Predicha 2
Real 0            18           0           0
Real 1             1          20           0
Real 2             0           0          15
```

La segunda capa no mejora accuracy ni F1 en esta ejecución. Su log-loss de
prueba es ligeramente menor, pero eso no demuestra una mejora general:
el log-loss medio de CV con la tasa elegida es 0,03430 para la red anterior
y 0,04088 para la nueva. Ambos resultados deben leerse junto a la variación
por pliegue y al tamaño pequeño de la prueba.

### Comparación de tasas

Todas las combinaciones obtienen el mismo F1 macro medio de CV, 98,34 %.
Se elige `0.001` en las dos arquitecturas por la regla de empate exacto, no
porque exista evidencia de superioridad de esa tasa.

| Tasa inicial | Épocas medias CV, red anterior | Épocas medias CV, red nueva |
|---:|---:|---:|
| 0.001 | 371,0 | 405,4 |
| 0.01 | 102,6 | 108,0 |
| 0.1 | 33,2 | 37,2 |

Ningún fold alcanzó el máximo de 2 000 épocas. Las tasas mayores necesitaron
menos épocas, aunque ese dato por sí solo no permite elegir un mejor modelo.

## Figuras y fronteras

- `mlp_wine_comparacion.png`: métricas de los dos modelos principales y
  matrices de confusión.
- `mlp_wine_learning_rates.png`: comparación de tasas por CV y curvas del
  costo de entrenamiento.
- `mlp_wine_fronteras.png`: regiones de decisión de dos modelos auxiliares
  que usan alcohol y flavanoides.

Las fronteras se ajustan con los mismos índices de entrenamiento/prueba y
la tasa seleccionada para cada arquitectura, pero emplean solo dos
atributos. Sus métricas se guardan separadas como `frontera_2d`. No son una
representación exacta de los clasificadores principales de 13 entradas.

En estos modelos auxiliares, la red de una capa obtiene accuracy 88,89 %,
F1 macro 89,11 % y log-loss 0,27245. La red de dos capas obtiene accuracy
90,74 %, F1 macro 90,99 % y log-loss 0,31597. Acertar más etiquetas y
empeorar log-loss es posible: esta última métrica también evalúa las
probabilidades asignadas a cada clase.

![Métricas y matrices de confusión](outputs/mlp_wine_comparacion.png)

![Tasas de aprendizaje y curvas de costo](outputs/mlp_wine_learning_rates.png)

![Fronteras de decisión auxiliares](outputs/mlp_wine_fronteras.png)

## Exportación de resultados

Las ejecuciones escriben en `output/`. Los archivos de referencia del
repositorio están en `outputs/`.

| CSV | Contenido y filas predeterminadas |
|---|---|
| `mlp_wine_metricas.csv` | Entrenamiento, prueba y frontera 2D por modelo: 6 filas |
| `mlp_wine_cv.csv` | Dos modelos × tres tasas × cinco pliegues: 30 filas |
| `mlp_wine_learning_rates.csv` | Resumen de CV por modelo y tasa: 6 filas |
| `mlp_wine_confusion.csv` | Dos matrices de tres por tres: 18 filas |
| `mlp_wine_costos.csv` | Costo por época de cada modelo y tasa |
| `mlp_wine_predicciones.csv` | Las 54 pruebas de cada modelo principal: 108 filas |
| `mlp_wine_parametros.csv` | Pesos y sesgos finales seleccionados: 275 + 387 = 662 filas |
| `mlp_wine_escalado.csv` | Media y escala de 13 atributos por modelo: 26 filas |

El archivo de parámetros usa `modelo`, `capa`, `tipo`, `origen`, `destino`
y `valor`. La columna `capa` distingue neuronas ocultas con nombres como
`h1` en capas diferentes. Solo exporta parámetros de los dos modelos
principales seleccionados, no de la CV ni de las fronteras auxiliares.

## Opciones

| Opción | Valor inicial | Efecto |
|---|---|---|
| `--learning-rates` | `0.001 0.01 0.1` | Una o más tasas positivas candidatas |
| `--max-iter` | `2000` | Máximo de épocas por ajuste |
| `--cv-folds` | `5` | Número de pliegues estratificados |
| `--seed` | `42` | Semilla de partición, CV e inicialización |
| `--output-dir` | `output` | Carpeta de exportación |
| `--no-gui` | Desactivado | Guardar sin abrir ventanas |

Las dos arquitecturas son fijas en este ejercicio. Use `--help` para ver la
ayuda. Un `ConvergenceWarning` indica que se llegó al límite de épocas sin
cumplir el criterio de parada; revise las curvas antes de interpretar
la comparación.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mlp_wine.py -v
```

Las seis pruebas del ejercicio verifican los datos y folds compartidos,
arquitecturas y activaciones, escalado sin fuga, elección de tasa y desempate,
recálculo de métricas, consistencia con la red anterior y exportación de los
ocho CSV. También reconstruyen las probabilidades a partir de los pesos,
sesgos y escalado exportados. La batería completa del repositorio quedó en
**62/62 pruebas superadas**.

## Alcance de las conclusiones

La prueba tiene solo 54 muestras: un error cambia el accuracy unos 1,85
puntos porcentuales. Además, ese mismo conjunto ya se examinó en el
ejercicio anterior. Esta comparación es exploratoria; no constituye una
evaluación en un benchmark nunca observado.

La CV selecciona hiperparámetros, por lo que su mejor valor puede resultar
optimista. Una segunda capa puede mejorar, empatar o empeorar el desempeño.
Para fundamentar una conclusión general conviene repetir varias semillas,
emplear CV anidada y conservar una prueba externa que no intervenga en las
decisiones. También se pueden estudiar saturación de la sigmoide,
regularización y parada temprana.
