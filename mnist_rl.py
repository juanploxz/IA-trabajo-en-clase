"""Reconocimiento de MNIST mediante una política de aprendizaje por refuerzo.

Cada imagen es el contexto de un bandido contextual. Las diez acciones son los
dígitos posibles y REINFORCE ajusta una política softmax con recompensa +1 por
acierto y -1 por error. El archivo también permite consultar MNIST y reconocer
imágenes externas desde la terminal o una interfaz Tkinter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path
import struct
import sys
from time import perf_counter
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DIRECTORIO_DATOS = Path("data") / "mnist"
RUTA_MODELO = Path("models") / "politica_mnist_rl.npz"
URL_BASE_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist"

# Nombre remoto, MD5 publicado y nombre de la caché NumPy.
ARCHIVOS_MNIST = {
    "imagenes_entrenamiento": (
        "train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "imagenes_entrenamiento.npy",
        True,
    ),
    "etiquetas_entrenamiento": (
        "train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
        "etiquetas_entrenamiento.npy",
        False,
    ),
    "imagenes_prueba": (
        "t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
        "imagenes_prueba.npy",
        True,
    ),
    "etiquetas_prueba": (
        "t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
        "etiquetas_prueba.npy",
        False,
    ),
}


@dataclass(slots=True)
class DatosMNIST:
    """Arreglos de MNIST; las imágenes se cargan mediante memoria mapeada."""

    imagenes_entrenamiento: np.ndarray
    etiquetas_entrenamiento: np.ndarray
    imagenes_prueba: np.ndarray
    etiquetas_prueba: np.ndarray


def _md5(ruta: Path) -> str:
    resumen = hashlib.md5(usedforsecurity=False)
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def _descargar_archivo(nombre: str, md5_esperado: str, directorio: Path) -> Path:
    """Descarga un archivo de forma segura y comprueba su integridad."""
    destino = directorio / nombre
    if destino.is_file() and _md5(destino) == md5_esperado:
        print(f"Disponible: {destino}")
        return destino

    directorio.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".part")
    url = f"{URL_BASE_MNIST}/{nombre}"
    print(f"Descargando {url}")
    solicitud = urllib.request.Request(
        url,
        headers={"User-Agent": "MNIST-RL-Academico/1.0"},
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=90) as respuesta:
            total = int(respuesta.headers.get("Content-Length", 0))
            descargado = 0
            with temporal.open("wb") as archivo:
                while True:
                    bloque = respuesta.read(1024 * 1024)
                    if not bloque:
                        break
                    archivo.write(bloque)
                    descargado += len(bloque)
                    if total:
                        porcentaje = descargado * 100.0 / total
                        print(
                            f"\r  {porcentaje:5.1f}%",
                            end="",
                            flush=True,
                        )
            if total:
                print()
    except (OSError, urllib.error.URLError) as error:
        if temporal.exists():
            temporal.unlink()
        raise RuntimeError(
            "No fue posible descargar MNIST. Compruebe la conexión a internet "
            f"y vuelva a ejecutar el comando: {error}"
        ) from error

    if _md5(temporal) != md5_esperado:
        temporal.unlink()
        raise RuntimeError(f"La verificación MD5 falló para {nombre}.")
    temporal.replace(destino)
    return destino


def _leer_imagenes_idx(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as archivo:
        cabecera = archivo.read(16)
        if len(cabecera) != 16:
            raise ValueError(f"Cabecera IDX incompleta en {ruta}.")
        magia, cantidad, filas, columnas = struct.unpack(">IIII", cabecera)
        if magia != 2051 or (filas, columnas) != (28, 28):
            raise ValueError(f"Formato de imágenes MNIST inválido en {ruta}.")
        contenido = archivo.read()
    esperado = cantidad * filas * columnas
    if len(contenido) != esperado:
        raise ValueError(f"Cantidad de píxeles incorrecta en {ruta}.")
    return np.frombuffer(contenido, dtype=np.uint8).reshape(
        cantidad,
        filas,
        columnas,
    )


def _leer_etiquetas_idx(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as archivo:
        cabecera = archivo.read(8)
        if len(cabecera) != 8:
            raise ValueError(f"Cabecera IDX incompleta en {ruta}.")
        magia, cantidad = struct.unpack(">II", cabecera)
        if magia != 2049:
            raise ValueError(f"Formato de etiquetas MNIST inválido en {ruta}.")
        contenido = archivo.read()
    if len(contenido) != cantidad:
        raise ValueError(f"Cantidad de etiquetas incorrecta en {ruta}.")
    return np.frombuffer(contenido, dtype=np.uint8)


def preparar_mnist(directorio: Path = DIRECTORIO_DATOS) -> None:
    """Descarga IDX, verifica MD5 y crea cachés NPY de acceso rápido."""
    directorio.mkdir(parents=True, exist_ok=True)
    for nombre, md5_esperado, cache, contiene_imagenes in ARCHIVOS_MNIST.values():
        ruta_cache = directorio / cache
        if ruta_cache.is_file():
            print(f"Caché disponible: {ruta_cache}")
            continue
        ruta_gzip = _descargar_archivo(nombre, md5_esperado, directorio)
        print(f"Preparando {ruta_cache}...")
        arreglo = (
            _leer_imagenes_idx(ruta_gzip)
            if contiene_imagenes
            else _leer_etiquetas_idx(ruta_gzip)
        )
        np.save(ruta_cache, arreglo, allow_pickle=False)
    print("MNIST está preparado para acceso mediante memoria mapeada.")


def cargar_mnist(directorio: Path = DIRECTORIO_DATOS) -> DatosMNIST:
    """Carga MNIST rápidamente; prepara los archivos si todavía no existen."""
    caches = [directorio / datos[2] for datos in ARCHIVOS_MNIST.values()]
    if not all(ruta.is_file() for ruta in caches):
        preparar_mnist(directorio)
    arreglos = {
        clave: np.load(directorio / datos[2], mmap_mode="r", allow_pickle=False)
        for clave, datos in ARCHIVOS_MNIST.items()
    }
    datos_mnist = DatosMNIST(**arreglos)
    if datos_mnist.imagenes_entrenamiento.shape != (60_000, 28, 28):
        raise ValueError("El conjunto de entrenamiento MNIST está incompleto.")
    if datos_mnist.imagenes_prueba.shape != (10_000, 28, 28):
        raise ValueError("El conjunto de prueba MNIST está incompleto.")
    return datos_mnist


class PoliticaMNISTRL:
    """Política softmax lineal entrenada con gradiente REINFORCE y Adam."""

    def __init__(self, pesos: np.ndarray, sesgo: np.ndarray) -> None:
        if pesos.shape != (784, 10) or sesgo.shape != (10,):
            raise ValueError("El modelo debe tener pesos (784, 10) y sesgo (10,).")
        self.pesos = pesos.astype(np.float32, copy=True)
        self.sesgo = sesgo.astype(np.float32, copy=True)

    @classmethod
    def crear(cls, semilla: int = 42) -> PoliticaMNISTRL:
        """Inicializa una política casi uniforme."""
        generador = np.random.default_rng(semilla)
        pesos = generador.normal(0.0, 0.005, size=(784, 10)).astype(np.float32)
        return cls(pesos, np.zeros(10, dtype=np.float32))

    @staticmethod
    def _normalizar(imagenes: np.ndarray) -> np.ndarray:
        return imagenes.reshape(len(imagenes), 784).astype(np.float32) / 255.0

    def probabilidades(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula π(a|s) de forma numéricamente estable."""
        logits = entradas @ self.pesos + self.sesgo
        logits -= logits.max(axis=1, keepdims=True)
        exponenciales = np.exp(logits)
        return exponenciales / exponenciales.sum(axis=1, keepdims=True)

    def predecir_lote(self, imagenes: np.ndarray) -> np.ndarray:
        entradas = self._normalizar(imagenes)
        return np.argmax(self.probabilidades(entradas), axis=1)

    def predecir_imagen(self, imagen: np.ndarray) -> tuple[int, np.ndarray]:
        if imagen.shape != (28, 28):
            raise ValueError("La imagen procesada debe tener dimensiones 28x28.")
        entradas = imagen.reshape(1, 784).astype(np.float32) / 255.0
        probabilidades = self.probabilidades(entradas)[0]
        return int(np.argmax(probabilidades)), probabilidades

    def exactitud(
        self,
        imagenes: np.ndarray,
        etiquetas: np.ndarray,
        tamano_bloque: int = 2_000,
    ) -> float:
        """Evalúa por bloques para limitar el uso de memoria."""
        correctas = 0
        for inicio in range(0, len(imagenes), tamano_bloque):
            fin = min(inicio + tamano_bloque, len(imagenes))
            predicciones = self.predecir_lote(imagenes[inicio:fin])
            correctas += int(np.sum(predicciones == etiquetas[inicio:fin]))
        return correctas / len(imagenes)

    def entrenar(
        self,
        imagenes: np.ndarray,
        etiquetas: np.ndarray,
        imagenes_prueba: np.ndarray,
        etiquetas_prueba: np.ndarray,
        epocas: int = 10,
        tamano_lote: int = 512,
        tasa_aprendizaje: float = 0.01,
        semilla: int = 42,
        limite: int | None = None,
    ) -> list[dict[str, float]]:
        """Entrena con REINFORCE usando la recompensa como única señal."""
        if epocas <= 0 or tamano_lote <= 0 or tasa_aprendizaje <= 0.0:
            raise ValueError("Épocas, lote y tasa de aprendizaje deben ser positivos.")
        cantidad = len(imagenes) if limite is None else min(limite, len(imagenes))
        if cantidad <= 0:
            raise ValueError("El límite de imágenes debe ser positivo.")

        generador = np.random.default_rng(semilla)
        momento_pesos = np.zeros_like(self.pesos)
        velocidad_pesos = np.zeros_like(self.pesos)
        momento_sesgo = np.zeros_like(self.sesgo)
        velocidad_sesgo = np.zeros_like(self.sesgo)
        beta_1, beta_2 = 0.9, 0.999
        paso_adam = 0
        linea_base = -0.8
        historial: list[dict[str, float]] = []

        for epoca in range(1, epocas + 1):
            inicio_epoca = perf_counter()
            indices = generador.permutation(len(imagenes))[:cantidad]
            suma_recompensas = 0.0
            acciones_correctas = 0

            for inicio in range(0, cantidad, tamano_lote):
                indices_lote = indices[inicio : inicio + tamano_lote]
                entradas = self._normalizar(imagenes[indices_lote])
                objetivos = np.asarray(etiquetas[indices_lote], dtype=np.int64)
                probabilidades = self.probabilidades(entradas)

                acumuladas = np.cumsum(probabilidades, axis=1)
                sorteos = generador.random((len(indices_lote), 1))
                acciones = np.sum(sorteos > acumuladas, axis=1)
                acciones = np.minimum(acciones, 9)
                recompensas = np.where(acciones == objetivos, 1.0, -1.0)
                suma_recompensas += float(recompensas.sum())
                acciones_correctas += int(np.sum(acciones == objetivos))

                media_lote = float(recompensas.mean())
                linea_base = 0.95 * linea_base + 0.05 * media_lote
                ventajas = recompensas - linea_base

                gradiente_logits = -probabilidades
                gradiente_logits[
                    np.arange(len(indices_lote)),
                    acciones,
                ] += 1.0
                gradiente_logits *= ventajas[:, None]
                gradiente_logits /= len(indices_lote)

                # Mantiene exploración al inicio y evita colapsos prematuros.
                log_prob = np.log(np.clip(probabilidades, 1e-8, 1.0))
                entropia = -np.sum(
                    probabilidades * log_prob,
                    axis=1,
                    keepdims=True,
                )
                gradiente_logits += 0.002 * (
                    -probabilidades * (log_prob + entropia)
                ) / len(indices_lote)

                gradiente_pesos = entradas.T @ gradiente_logits
                gradiente_pesos -= 1e-5 * self.pesos
                gradiente_sesgo = gradiente_logits.sum(axis=0)

                norma = float(
                    np.sqrt(
                        np.sum(gradiente_pesos**2)
                        + np.sum(gradiente_sesgo**2)
                    )
                )
                if norma > 5.0:
                    factor = 5.0 / norma
                    gradiente_pesos *= factor
                    gradiente_sesgo *= factor

                paso_adam += 1
                momento_pesos = (
                    beta_1 * momento_pesos + (1.0 - beta_1) * gradiente_pesos
                )
                velocidad_pesos = (
                    beta_2 * velocidad_pesos
                    + (1.0 - beta_2) * gradiente_pesos**2
                )
                momento_sesgo = (
                    beta_1 * momento_sesgo + (1.0 - beta_1) * gradiente_sesgo
                )
                velocidad_sesgo = (
                    beta_2 * velocidad_sesgo
                    + (1.0 - beta_2) * gradiente_sesgo**2
                )
                correccion_1 = 1.0 - beta_1**paso_adam
                correccion_2 = 1.0 - beta_2**paso_adam
                self.pesos += tasa_aprendizaje * (
                    momento_pesos / correccion_1
                ) / (np.sqrt(velocidad_pesos / correccion_2) + 1e-8)
                self.sesgo += tasa_aprendizaje * (
                    momento_sesgo / correccion_1
                ) / (np.sqrt(velocidad_sesgo / correccion_2) + 1e-8)

            exactitud_prueba = self.exactitud(
                imagenes_prueba,
                etiquetas_prueba,
            )
            recompensa_media = suma_recompensas / cantidad
            tasa_aciertos_muestreados = acciones_correctas / cantidad
            duracion = perf_counter() - inicio_epoca
            registro = {
                "epoca": float(epoca),
                "recompensa_media": recompensa_media,
                "aciertos_muestreados": tasa_aciertos_muestreados,
                "exactitud_prueba": exactitud_prueba,
                "segundos": duracion,
            }
            historial.append(registro)
            print(
                f"Época {epoca:02d}/{epocas:02d} | "
                f"recompensa={recompensa_media:+.4f} | "
                f"acciones correctas={tasa_aciertos_muestreados:.2%} | "
                f"exactitud test={exactitud_prueba:.2%} | {duracion:.2f} s"
            )
        return historial

    def guardar(self, ruta: Path = RUTA_MODELO) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ruta,
            pesos=self.pesos,
            sesgo=self.sesgo,
            version=np.array([1], dtype=np.int16),
        )

    @classmethod
    def cargar(cls, ruta: Path = RUTA_MODELO) -> PoliticaMNISTRL:
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe el modelo {ruta}. Ejecute primero el comando entrenar."
            )
        with np.load(ruta, allow_pickle=False) as archivo:
            pesos = archivo["pesos"]
            sesgo = archivo["sesgo"]
        return cls(pesos, sesgo)


def _desplazar_sin_envolver(
    imagen: np.ndarray,
    desplazamiento_x: int,
    desplazamiento_y: int,
) -> np.ndarray:
    resultado = np.zeros_like(imagen)
    origen_x = max(0, -desplazamiento_x)
    destino_x = max(0, desplazamiento_x)
    ancho = imagen.shape[1] - abs(desplazamiento_x)
    origen_y = max(0, -desplazamiento_y)
    destino_y = max(0, desplazamiento_y)
    alto = imagen.shape[0] - abs(desplazamiento_y)
    if ancho > 0 and alto > 0:
        resultado[
            destino_y : destino_y + alto,
            destino_x : destino_x + ancho,
        ] = imagen[
            origen_y : origen_y + alto,
            origen_x : origen_x + ancho,
        ]
    return resultado


def normalizar_imagen_externa(imagen: Image.Image) -> np.ndarray:
    """Convierte una imagen dibujada por el usuario al estilo MNIST 28x28."""
    escala_grises = ImageOps.grayscale(imagen)
    arreglo = np.asarray(escala_grises, dtype=np.uint8)
    bordes = np.concatenate(
        (arreglo[0], arreglo[-1], arreglo[:, 0], arreglo[:, -1])
    )
    if float(np.median(bordes)) > 127.0:
        arreglo = 255 - arreglo

    coordenadas = np.argwhere(arreglo > 30)
    if len(coordenadas) == 0:
        raise ValueError("La imagen no contiene un trazo reconocible.")
    (min_y, min_x), (max_y, max_x) = coordenadas.min(0), coordenadas.max(0)
    recorte = Image.fromarray(
        arreglo[min_y : max_y + 1, min_x : max_x + 1]
    )
    escala = min(20.0 / recorte.width, 20.0 / recorte.height)
    nuevo_tamano = (
        max(1, round(recorte.width * escala)),
        max(1, round(recorte.height * escala)),
    )
    recorte = recorte.resize(nuevo_tamano, Image.Resampling.LANCZOS)
    lienzo = Image.new("L", (28, 28), color=0)
    posicion = ((28 - recorte.width) // 2, (28 - recorte.height) // 2)
    lienzo.paste(recorte, posicion)
    procesada = np.asarray(lienzo, dtype=np.uint8)

    masa = float(procesada.sum())
    if masa > 0.0:
        indices_y, indices_x = np.indices(procesada.shape)
        centro_x = float(np.sum(indices_x * procesada) / masa)
        centro_y = float(np.sum(indices_y * procesada) / masa)
        procesada = _desplazar_sin_envolver(
            procesada,
            round(13.5 - centro_x),
            round(13.5 - centro_y),
        )
    return procesada


def cargar_imagen_externa(ruta: Path) -> np.ndarray:
    try:
        with Image.open(ruta) as imagen:
            return normalizar_imagen_externa(imagen)
    except OSError as error:
        raise ValueError(f"No fue posible abrir la imagen {ruta}: {error}") from error


def seleccionar_digito(
    imagenes: np.ndarray,
    etiquetas: np.ndarray,
    digito: int,
    cantidad: int,
    semilla: int,
) -> np.ndarray:
    """Devuelve índices reproducibles que contienen el dígito solicitado."""
    if not 0 <= digito <= 9:
        raise ValueError("El dígito debe estar entre 0 y 9.")
    if cantidad <= 0:
        raise ValueError("La cantidad debe ser positiva.")
    disponibles = np.flatnonzero(etiquetas == digito)
    if cantidad > len(disponibles):
        raise ValueError("La cantidad solicitada supera los ejemplos disponibles.")
    generador = np.random.default_rng(semilla)
    return generador.choice(disponibles, size=cantidad, replace=False)


def construir_mosaico(
    imagenes: np.ndarray,
    indices: np.ndarray,
) -> Image.Image:
    """Construye una cuadrícula PIL con imágenes e índices de MNIST."""
    columnas = math.ceil(math.sqrt(len(indices)))
    filas = math.ceil(len(indices) / columnas)
    ancho_celda, alto_celda = 72, 82
    mosaico = Image.new(
        "RGB",
        (columnas * ancho_celda, filas * alto_celda),
        "white",
    )
    dibujo = ImageDraw.Draw(mosaico)
    for posicion, indice in enumerate(indices):
        fila, columna = divmod(posicion, columnas)
        x, y = columna * ancho_celda, fila * alto_celda
        muestra = Image.fromarray(np.asarray(imagenes[int(indice)]))
        muestra = muestra.resize((64, 64), Image.Resampling.NEAREST).convert("RGB")
        mosaico.paste(muestra, (x + 4, y + 2))
        dibujo.text((x + 5, y + 67), f"índice {int(indice)}", fill="black")
    return mosaico


def guardar_consulta(
    imagenes: np.ndarray,
    indices: np.ndarray,
    ruta: Path,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    construir_mosaico(imagenes, indices).save(ruta)


def iniciar_interfaz(directorio: Path, ruta_modelo: Path) -> None:
    """Abre una interfaz para cargar imágenes y consultar MNIST por número."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except ImportError as error:
        raise RuntimeError(
            "Tkinter no está disponible. Use los comandos predecir y consultar."
        ) from error

    try:
        raiz = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "No se pudo abrir una ventana. Use predecir o consultar desde terminal."
        ) from error

    raiz.title("MNIST con Reinforcement Learning")
    raiz.geometry("780x680")
    raiz.minsize(700, 600)
    modelo = PoliticaMNISTRL.cargar(ruta_modelo) if ruta_modelo.is_file() else None
    datos_cargados: DatosMNIST | None = None

    marco = ttk.Frame(raiz, padding=16)
    marco.pack(fill="both", expand=True)
    ttk.Label(
        marco,
        text="Reconocimiento MNIST mediante una política REINFORCE",
        font=("Segoe UI", 16, "bold"),
    ).pack(pady=(0, 8))
    estado = tk.StringVar(
        value=(
            "Modelo listo. Cargue una imagen."
            if modelo is not None
            else "Modelo no entrenado. Ejecute: python mnist_rl.py entrenar"
        )
    )
    ttk.Label(marco, textvariable=estado, wraplength=720).pack(pady=(0, 10))
    etiqueta_imagen = ttk.Label(marco, anchor="center")
    etiqueta_imagen.pack(fill="both", expand=True, pady=8)

    def mostrar_imagen(imagen: Image.Image) -> None:
        copia = imagen.copy()
        copia.thumbnail((520, 420), Image.Resampling.NEAREST)
        foto = ImageTk.PhotoImage(copia)
        etiqueta_imagen.configure(image=foto)
        etiqueta_imagen.image = foto

    def reconocer() -> None:
        if modelo is None:
            messagebox.showerror(
                "Modelo ausente",
                "Entrene primero con: python mnist_rl.py entrenar",
            )
            return
        nombre = filedialog.askopenfilename(
            title="Seleccione una imagen con un dígito",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not nombre:
            return
        try:
            procesada = cargar_imagen_externa(Path(nombre))
            prediccion, probabilidades = modelo.predecir_imagen(procesada)
            mejores = np.argsort(probabilidades)[::-1][:3]
            detalle = ", ".join(
                f"{int(digito)}: {probabilidades[digito]:.1%}"
                for digito in mejores
            )
            estado.set(f"Predicción: {prediccion} | alternativas: {detalle}")
            mostrar_imagen(
                Image.fromarray(procesada).resize(
                    (336, 336),
                    Image.Resampling.NEAREST,
                )
            )
        except (ValueError, OSError) as error:
            messagebox.showerror("No se pudo analizar", str(error))

    controles = ttk.Frame(marco)
    controles.pack(fill="x", pady=8)
    ttk.Button(
        controles,
        text="Cargar imagen y reconocer",
        command=reconocer,
    ).pack(side="left", padx=4)
    ttk.Label(controles, text="Número MNIST:").pack(side="left", padx=(18, 4))
    numero = tk.StringVar(value="7")
    ttk.Spinbox(
        controles,
        from_=0,
        to=9,
        width=4,
        textvariable=numero,
    ).pack(side="left")

    def consultar() -> None:
        nonlocal datos_cargados
        try:
            digito = int(numero.get())
            estado.set("Cargando la base MNIST...")
            raiz.update_idletasks()
            if datos_cargados is None:
                datos_cargados = cargar_mnist(directorio)
            indices = seleccionar_digito(
                datos_cargados.imagenes_prueba,
                datos_cargados.etiquetas_prueba,
                digito,
                16,
                42,
            )
            mosaico = construir_mosaico(
                datos_cargados.imagenes_prueba,
                indices,
            )
            mostrar_imagen(mosaico)
            estado.set(
                f"16 ejemplos del número {digito}. Índices: "
                + ", ".join(str(int(indice)) for indice in indices)
            )
        except (ValueError, RuntimeError, OSError) as error:
            messagebox.showerror("Consulta fallida", str(error))

    ttk.Button(
        controles,
        text="Buscar en MNIST",
        command=consultar,
    ).pack(side="left", padx=8)
    raiz.mainloop()


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena y utiliza una política RL para reconocer dígitos MNIST."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DIRECTORIO_DATOS,
        help="Directorio de MNIST (predeterminado: data/mnist).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=RUTA_MODELO,
        help="Archivo de la política (predeterminado: models/politica_mnist_rl.npz).",
    )
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("preparar", help="Descarga y prepara MNIST.")

    entrenar = subparsers.add_parser("entrenar", help="Entrena la política RL.")
    entrenar.add_argument("--epochs", type=int, default=10)
    entrenar.add_argument("--batch-size", type=int, default=512)
    entrenar.add_argument("--learning-rate", type=float, default=0.01)
    entrenar.add_argument("--seed", type=int, default=42)
    entrenar.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita las imágenes por época para una demostración rápida.",
    )
    entrenar.add_argument(
        "--continuar",
        action="store_true",
        help="Continúa desde el modelo guardado, si existe.",
    )

    predecir = subparsers.add_parser(
        "predecir",
        help="Reconoce una o más imágenes externas.",
    )
    predecir.add_argument("imagenes", type=Path, nargs="+")

    consultar = subparsers.add_parser(
        "consultar",
        help="Busca rápidamente ejemplos de un número en MNIST.",
    )
    consultar.add_argument("digito", type=int)
    consultar.add_argument("--count", type=int, default=16)
    consultar.add_argument("--seed", type=int, default=42)
    consultar.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
    )
    consultar.add_argument("--output", type=Path, default=None)
    subparsers.add_parser("gui", help="Abre la interfaz gráfica.")
    return parser


def ejecutar(argumentos: argparse.Namespace) -> int:
    if argumentos.comando == "preparar":
        preparar_mnist(argumentos.data_dir)
        return 0
    if argumentos.comando == "entrenar":
        datos = cargar_mnist(argumentos.data_dir)
        if argumentos.continuar and argumentos.model.is_file():
            modelo = PoliticaMNISTRL.cargar(argumentos.model)
            print(f"Continuando desde {argumentos.model}")
        else:
            modelo = PoliticaMNISTRL.crear(argumentos.seed)
        historial = modelo.entrenar(
            datos.imagenes_entrenamiento,
            datos.etiquetas_entrenamiento,
            datos.imagenes_prueba,
            datos.etiquetas_prueba,
            epocas=argumentos.epochs,
            tamano_lote=argumentos.batch_size,
            tasa_aprendizaje=argumentos.learning_rate,
            semilla=argumentos.seed,
            limite=argumentos.limit,
        )
        modelo.guardar(argumentos.model)
        print(f"Modelo guardado en: {argumentos.model.resolve()}")
        print(
            "Exactitud final en 10 000 imágenes de prueba: "
            f"{historial[-1]['exactitud_prueba']:.2%}"
        )
        return 0
    if argumentos.comando == "predecir":
        modelo = PoliticaMNISTRL.cargar(argumentos.model)
        for ruta in argumentos.imagenes:
            imagen = cargar_imagen_externa(ruta)
            prediccion, probabilidades = modelo.predecir_imagen(imagen)
            mejores = np.argsort(probabilidades)[::-1][:3]
            alternativas = ", ".join(
                f"{int(digito)}={probabilidades[digito]:.2%}"
                for digito in mejores
            )
            print(
                f"{ruta}: predicción {prediccion} | política: {alternativas}"
            )
        return 0
    if argumentos.comando == "consultar":
        datos = cargar_mnist(argumentos.data_dir)
        if argumentos.split == "train":
            imagenes = datos.imagenes_entrenamiento
            etiquetas = datos.etiquetas_entrenamiento
        else:
            imagenes = datos.imagenes_prueba
            etiquetas = datos.etiquetas_prueba
        indices = seleccionar_digito(
            imagenes,
            etiquetas,
            argumentos.digito,
            argumentos.count,
            argumentos.seed,
        )
        salida = argumentos.output or (
            Path("output") / f"mnist_consulta_{argumentos.digito}.png"
        )
        guardar_consulta(imagenes, indices, salida)
        print(f"Índices encontrados: {[int(indice) for indice in indices]}")
        print(f"Mosaico guardado en: {salida.resolve()}")
        return 0
    if argumentos.comando == "gui":
        iniciar_interfaz(argumentos.data_dir, argumentos.model)
        return 0
    raise ValueError(f"Comando desconocido: {argumentos.comando}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argumentos = crear_parser().parse_args()
    try:
        return ejecutar(argumentos)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
