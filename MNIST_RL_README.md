# MNIST con aprendizaje por refuerzo

`mnist_rl.py` es una aplicación independiente del proyecto GridWorld. Reconoce
dígitos, permite buscar ejemplos por número en la base MNIST y ofrece una
interfaz para cargar imágenes.

## Formulación RL

MNIST normalmente se resuelve mejor mediante aprendizaje supervisado. En esta
demostración se formula como un **bandido contextual**:

- Estado o contexto: los 784 píxeles de una imagen de 28×28.
- Acciones: seleccionar uno de los dígitos del 0 al 9.
- Política: distribución softmax `π(a|s)`.
- Recompensa: `+1` si la acción coincide con la etiqueta y `-1` si falla.
- Algoritmo: REINFORCE con línea base, regularización de entropía y Adam.

La etiqueta solo se usa para calcular la recompensa. La actualización que
maximiza la recompensa esperada es:

```text
θ ← θ + α (R - b) ∇θ log πθ(a|s)
```

## Preparar la base de datos

```powershell
.\.venv\Scripts\python.exe mnist_rl.py preparar
```

Se descargan los cuatro archivos IDX, se verifica su MD5 y se crean archivos
`.npy`. Las consultas posteriores usan memoria mapeada y no necesitan cargar
toda la base en RAM.

## Entrenar

```powershell
.\.venv\Scripts\python.exe mnist_rl.py entrenar --epochs 10
```

El modelo queda en `models/politica_mnist_rl.npz`. Para una demostración rápida:

```powershell
.\.venv\Scripts\python.exe mnist_rl.py entrenar --epochs 3 --limit 10000
```

Para continuar un modelo existente:

```powershell
.\.venv\Scripts\python.exe mnist_rl.py entrenar --epochs 3 --continuar
```

## Reconocer una imagen subida

La imagen puede ser PNG, JPG, BMP o GIF, con dígito oscuro sobre fondo claro o
viceversa. Se recorta, escala, centra y convierte automáticamente a 28×28.

```powershell
.\.venv\Scripts\python.exe mnist_rl.py predecir ruta\numero.png
```

El repositorio incluye una imagen lista para probar el flujo:

```powershell
.\.venv\Scripts\python.exe mnist_rl.py predecir examples\numero_7.png
```

## Consultar MNIST por número

Este ejemplo busca 16 imágenes del número 7 y crea un mosaico:

```powershell
.\.venv\Scripts\python.exe mnist_rl.py consultar 7 --count 16
```

La salida predeterminada es `output/mnist_consulta_7.png`. Puede utilizar
`--split train`, `--seed` o `--output otra_ruta.png`.

## Interfaz gráfica

```powershell
.\.venv\Scripts\python.exe mnist_rl.py gui
```

La interfaz tiene dos operaciones:

1. Cargar una imagen local y mostrar la predicción con las tres acciones más
   probables de la política.
2. Escribir un número entre 0 y 9 y recuperar rápidamente 16 ejemplos de MNIST.

## Pruebas

Las pruebas del archivo se incluyen en la batería general:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
