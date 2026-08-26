# Comparación de particiones en Iris

Este es un ejercicio independiente que estudia qué ocurre al cambiar la
cantidad de datos destinada al entrenamiento y a la prueba. Compara:

- 60 % entrenamiento y 40 % prueba.
- 70 % entrenamiento y 30 % prueba.
- 80 % entrenamiento y 20 % prueba.

En las tres configuraciones se conservan LDA, K-NN con `k=3`, árbol de
decisión, estratificación y semilla `42`.

## Ejecutar

Mostrar las gráficas:

```powershell
.\.venv\Scripts\python.exe iris_split_comparison.py
```

Generar los archivos sin abrir ventanas:

```powershell
.\.venv\Scripts\python.exe iris_split_comparison.py --no-gui
```

## Qué cambia

| Partición | Entrenamiento | Prueba | Por clase al entrenar | Por clase al probar |
|---|---:|---:|---:|---:|
| 60/40 | 90 | 60 | 30 | 20 |
| 70/30 | 105 | 45 | 35 | 15 |
| 80/20 | 120 | 30 | 40 | 10 |

Al aumentar el porcentaje de entrenamiento, el modelo dispone de más ejemplos
para aprender. Al mismo tiempo, quedan menos ejemplos para medir su rendimiento,
por lo que la estimación de las métricas puede ser menos estable. Tener más
entrenamiento no garantiza que una única ejecución produzca una puntuación
monótonamente mayor: también cambia cuáles flores forman el conjunto de prueba.

## Diseño experimental

- Se usa una partición estratificada para mantener el mismo número de flores de
  cada especie dentro de cada conjunto.
- Los tres clasificadores reciben exactamente la misma partición en cada
  configuración.
- La semilla fija permite reproducir las observaciones.
- Accuracy, precisión macro, recall macro y F1 macro usan los cuatro atributos.
- Las fronteras se ajustan por separado con longitud y ancho del pétalo, porque
  una frontera de cuatro dimensiones no puede dibujarse en un plano.

## Archivos generados

```text
output/iris_particiones_metricas.png
output/iris_particiones_fronteras.png
output/iris_particiones_metricas.csv
output/iris_particiones_predicciones.csv
```

El primer CSV contiene nueve filas: tres particiones por tres clasificadores,
incluidas las cuatro métricas, el número de errores y las nueve celdas de la
matriz de confusión. El segundo contiene las 135 evaluaciones acumuladas:
60 + 45 + 30 muestras de prueba.

## Resultados verificados

Resultados obtenidos con semilla `42` y los cuatro atributos:

| Partición | Clasificador | Accuracy | F1 macro |
|---|---|---:|---:|
| 60/40 | LDA | 98,33 % | 98,33 % |
| 60/40 | K-NN (k=3) | 95,00 % | 94,97 % |
| 60/40 | Árbol de decisión | 95,00 % | 95,00 % |
| 70/30 | LDA | 97,78 % | 97,78 % |
| 70/30 | K-NN (k=3) | 91,11 % | 90,95 % |
| 70/30 | Árbol de decisión | 93,33 % | 93,27 % |
| 80/20 | LDA | 100,00 % | 100,00 % |
| 80/20 | K-NN (k=3) | 93,33 % | 93,27 % |
| 80/20 | Árbol de decisión | 93,33 % | 93,33 % |

LDA alcanzó 100 % con 80/20, pero esa prueba solo contiene 30 flores. No debe
interpretarse como garantía de que 80/20 siempre sea mejor. K-NN y el árbol
obtuvieron su F1 más alto con 60/40, aun teniendo menos datos de entrenamiento.
Esto demuestra que el resultado también depende de qué observaciones quedan en
prueba y que una sola partición no basta para establecer superioridad general.

En la prueba 80/20, cada error modifica el accuracy en 1/30 = 3,33 puntos
porcentuales. En 60/40, cada error representa 1/60 = 1,67 puntos. Por tanto,
las métricas de 80/20 pueden cambiar más ante una única flor difícil.

## Argumentos

| Argumento | Descripción | Predeterminado |
|---|---|---:|
| `--seed N` | Semilla compartida | `42` |
| `--neighbors N` | Vecinos de K-NN | `3` |
| `--tree-max-depth N` | Profundidad máxima del árbol | Sin límite |
| `--features X Y` | Atributos visibles en las fronteras | `2 3` |
| `--output-dir RUTA` | Directorio de resultados | `output` |
| `--no-gui` | Guarda sin abrir ventanas | Desactivado |

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_iris_split_comparison -v
```

Las pruebas validan tamaños, estratificación, modelos, métricas,
reproducibilidad, filas de los CSV y generación real de las dos figuras.

## Interpretación académica

La comparación ilustra el compromiso entre aprendizaje y evaluación:

1. 60/40 ofrece una prueba más grande y una medición más estable, pero deja
   menos ejemplos para que los modelos aprendan.
2. 70/30 es un compromiso habitual entre ambos objetivos.
3. 80/20 entrega más información al modelo, pero cada error representa 3,33
   puntos porcentuales de accuracy porque la prueba solo contiene 30 flores.
4. Para una conclusión más general se recomienda repetir varias semillas o
   aplicar validación cruzada estratificada.
