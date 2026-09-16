# Ejercicio 8: 100 puntos sobre una recta con ruido

El programa independiente `recta_con_ruido.py` genera 100 coordenadas `(x, y)`
y dibuja una gráfica de dispersión junto a la recta ideal.

## Fórmula y datos

La ecuación correcta de la recta es **y = m*x + b**. Para añadir dispersión:

```text
y = m*x + b + ε
```

- `x`: 100 valores aleatorios uniformes entre 0 y 10.
- `m`: pendiente; inicialmente 2.
- `b`: intercepto, el valor ideal de `y` cuando `x=0`; inicialmente 1.
- `ε`: ruido normal de media 0 y desviación estándar 2 por defecto.

Por tanto, la configuración inicial produce puntos alrededor de `y = 2*x + 1`.
El ruido se añade solo a `y`. Su media teórica es cero, pero la media de una
muestra de 100 valores no tiene que ser exactamente cero.

## Ejecutar

Desde la carpeta del proyecto, con las dependencias ya instaladas:

```powershell
.\.venv\Scripts\python.exe recta_con_ruido.py
```

Para guardar la figura sin abrir una ventana:

```powershell
.\.venv\Scripts\python.exe recta_con_ruido.py --no-gui
```

En Linux o macOS, sustituya `.\.venv\Scripts\python.exe` por `./.venv/bin/python`.
Se utilizan NumPy y Matplotlib, incluidos en `requirements.txt`.

## Opciones

| Opción | Valor inicial | Significado |
|---|---:|---|
| `--m` | 2 | Pendiente de la recta |
| `--b` | 1 | Intercepto |
| `--noise` | 2 | Desviación estándar del ruido; debe ser no negativa |
| `--seed` | 42 | Semilla del generador aleatorio |
| `--output-dir` | `output` | Carpeta para la figura y los datos |
| `--no-gui` | Desactivado | Guarda los archivos sin abrir la gráfica |

Siempre se generan 100 puntos. La misma semilla y los mismos parámetros
reproducen la misma muestra; cambie `--seed` para obtener otra.

```powershell
# Otra pendiente e intercepto, con menos dispersión
.\.venv\Scripts\python.exe recta_con_ruido.py --m 3 --b -2 --noise 1

# Puntos exactamente sobre la recta
.\.venv\Scripts\python.exe recta_con_ruido.py --noise 0

# Mayor dispersión y otra muestra aleatoria
.\.venv\Scripts\python.exe recta_con_ruido.py --noise 5 --seed 7
```

## Archivos generados

- `output/recta_con_ruido.png`: puntos observados y recta ideal sin ruido.
- `output/recta_con_ruido.csv`: 100 filas, con columnas `x`, `y`, `y_sin_ruido`
  y `ruido`.

En cada fila se cumple `y_sin_ruido = m*x + b` y `y = y_sin_ruido + ruido`.
La línea representa la relación utilizada para generar los datos; no es una
recta estimada mediante entrenamiento. A mayor `--noise`, mayor dispersión
vertical esperada alrededor de ella.

Las nuevas ejecuciones guardan archivos con los mismos nombres en la carpeta
de salida seleccionada. Use otra carpeta con `--output-dir` para conservar
resultados de varias configuraciones.
