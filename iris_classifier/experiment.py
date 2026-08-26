"""Carga, partición, entrenamiento y evaluación del ejercicio Iris."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

NOMBRES_ATRIBUTOS_ES = (
    "Longitud del sépalo (cm)",
    "Ancho del sépalo (cm)",
    "Longitud del pétalo (cm)",
    "Ancho del pétalo (cm)",
)
NOMBRES_CLASES = ("Setosa", "Versicolor", "Virginica")


@dataclass(slots=True)
class ResultadoClasificador:
    """Modelo ajustado y métricas sobre el conjunto de prueba."""

    nombre: str
    modelo: Any
    predicciones: np.ndarray
    metricas: dict[str, float]
    matriz_confusion: np.ndarray
    reporte_clasificacion: str


@dataclass(slots=True)
class ResultadoFrontera:
    """Modelo bidimensional usado exclusivamente para dibujar su frontera."""

    nombre: str
    modelo: Any
    exactitud_2d: float


@dataclass(slots=True)
class ResultadoExperimento:
    """Datos, partición estratificada y resultados de los tres métodos."""

    datos: np.ndarray
    etiquetas: np.ndarray
    indices_entrenamiento: np.ndarray
    indices_prueba: np.ndarray
    x_entrenamiento: np.ndarray
    x_prueba: np.ndarray
    y_entrenamiento: np.ndarray
    y_prueba: np.ndarray
    nombres_atributos: tuple[str, ...]
    nombres_clases: tuple[str, ...]
    clasificadores: dict[str, ResultadoClasificador]
    proporcion_prueba: float
    semilla: int


def crear_clasificadores(
    vecinos: int = 3,
    profundidad_arbol: int | None = None,
    semilla: int = 42,
) -> dict[str, Any]:
    """Construye los estimadores sin ajustar con parámetros reproducibles."""
    if vecinos <= 0:
        raise ValueError("La cantidad de vecinos debe ser positiva.")
    if profundidad_arbol is not None and profundidad_arbol <= 0:
        raise ValueError("La profundidad del árbol debe ser positiva.")
    return {
        "LDA": LinearDiscriminantAnalysis(),
        f"K-NN (k={vecinos})": Pipeline(
            [
                ("escalador", StandardScaler()),
                ("clasificador", KNeighborsClassifier(n_neighbors=vecinos)),
            ]
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=profundidad_arbol,
            random_state=semilla,
        ),
    }


def ejecutar_experimento(
    proporcion_prueba: float = 0.30,
    semilla: int = 42,
    vecinos: int = 3,
    profundidad_arbol: int | None = None,
) -> ResultadoExperimento:
    """Aplica una única partición 70/30 estratificada a los tres modelos."""
    if not 0.0 < proporcion_prueba < 1.0:
        raise ValueError("La proporción de prueba debe pertenecer a (0, 1).")
    iris = load_iris()
    datos = np.asarray(iris.data, dtype=np.float64)
    etiquetas = np.asarray(iris.target, dtype=np.int64)
    indices = np.arange(len(datos), dtype=np.int64)

    (
        x_entrenamiento,
        x_prueba,
        y_entrenamiento,
        y_prueba,
        indices_entrenamiento,
        indices_prueba,
    ) = train_test_split(
        datos,
        etiquetas,
        indices,
        test_size=proporcion_prueba,
        random_state=semilla,
        stratify=etiquetas,
    )

    resultados: dict[str, ResultadoClasificador] = {}
    for nombre, modelo in crear_clasificadores(
        vecinos,
        profundidad_arbol,
        semilla,
    ).items():
        modelo.fit(x_entrenamiento, y_entrenamiento)
        predicciones = modelo.predict(x_prueba)
        metricas = {
            "accuracy": float(accuracy_score(y_prueba, predicciones)),
            "precision_macro": float(
                precision_score(
                    y_prueba,
                    predicciones,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    y_prueba,
                    predicciones,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_macro": float(
                f1_score(
                    y_prueba,
                    predicciones,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        resultados[nombre] = ResultadoClasificador(
            nombre=nombre,
            modelo=modelo,
            predicciones=np.asarray(predicciones, dtype=np.int64),
            metricas=metricas,
            matriz_confusion=confusion_matrix(
                y_prueba,
                predicciones,
                labels=np.arange(len(NOMBRES_CLASES)),
            ),
            reporte_clasificacion=classification_report(
                y_prueba,
                predicciones,
                target_names=NOMBRES_CLASES,
                zero_division=0,
            ),
        )

    return ResultadoExperimento(
        datos=datos,
        etiquetas=etiquetas,
        indices_entrenamiento=indices_entrenamiento,
        indices_prueba=indices_prueba,
        x_entrenamiento=x_entrenamiento,
        x_prueba=x_prueba,
        y_entrenamiento=y_entrenamiento,
        y_prueba=y_prueba,
        nombres_atributos=NOMBRES_ATRIBUTOS_ES,
        nombres_clases=NOMBRES_CLASES,
        clasificadores=resultados,
        proporcion_prueba=proporcion_prueba,
        semilla=semilla,
    )


def entrenar_modelos_frontera(
    resultado: ResultadoExperimento,
    atributos: tuple[int, int] = (2, 3),
) -> dict[str, ResultadoFrontera]:
    """Ajusta clones con dos variables para una frontera 2D interpretable."""
    if len(atributos) != 2 or atributos[0] == atributos[1]:
        raise ValueError("Las fronteras requieren dos atributos diferentes.")
    if any(indice not in range(resultado.datos.shape[1]) for indice in atributos):
        raise ValueError("El índice de atributo debe estar entre 0 y 3.")
    x_entrenamiento = resultado.x_entrenamiento[:, atributos]
    x_prueba = resultado.x_prueba[:, atributos]
    fronteras: dict[str, ResultadoFrontera] = {}
    for nombre, clasificador in resultado.clasificadores.items():
        modelo = clone(clasificador.modelo)
        modelo.fit(x_entrenamiento, resultado.y_entrenamiento)
        exactitud = accuracy_score(resultado.y_prueba, modelo.predict(x_prueba))
        fronteras[nombre] = ResultadoFrontera(
            nombre=nombre,
            modelo=modelo,
            exactitud_2d=float(exactitud),
        )
    return fronteras


def guardar_metricas_csv(resultado: ResultadoExperimento, ruta: Path) -> None:
    """Exporta una fila por clasificador para análisis posterior."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            ("clasificador", "accuracy", "precision_macro", "recall_macro", "f1_macro")
        )
        for clasificador in resultado.clasificadores.values():
            escritor.writerow(
                (
                    clasificador.nombre,
                    clasificador.metricas["accuracy"],
                    clasificador.metricas["precision_macro"],
                    clasificador.metricas["recall_macro"],
                    clasificador.metricas["f1_macro"],
                )
            )


def guardar_predicciones_csv(resultado: ResultadoExperimento, ruta: Path) -> None:
    """Exporta muestras de prueba, clase real y las tres predicciones."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_modelos = tuple(resultado.clasificadores)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "indice_iris",
                *resultado.nombres_atributos,
                "clase_real",
                *nombres_modelos,
            )
        )
        for posicion, indice_original in enumerate(resultado.indices_prueba):
            predicciones = tuple(
                resultado.clasificadores[nombre].predicciones[posicion]
                for nombre in nombres_modelos
            )
            escritor.writerow(
                (
                    int(indice_original),
                    *resultado.x_prueba[posicion].tolist(),
                    resultado.nombres_clases[int(resultado.y_prueba[posicion])],
                    *(
                        resultado.nombres_clases[int(prediccion)]
                        for prediccion in predicciones
                    ),
                )
            )


def imprimir_resumen(resultado: ResultadoExperimento) -> None:
    """Muestra partición, métricas y reportes en la terminal."""
    porcentaje_entrenamiento = (1.0 - resultado.proporcion_prueba) * 100.0
    porcentaje_prueba = resultado.proporcion_prueba * 100.0
    print("=== Clasificación del conjunto Iris ===")
    print(f"Muestras totales: {len(resultado.datos)}")
    print(
        f"Entrenamiento: {len(resultado.x_entrenamiento)} "
        f"({porcentaje_entrenamiento:.0f}%)"
    )
    print(f"Prueba: {len(resultado.x_prueba)} ({porcentaje_prueba:.0f}%)")
    print(f"Semilla: {resultado.semilla}")
    distribucion = np.bincount(
        resultado.y_prueba,
        minlength=len(resultado.nombres_clases),
    )
    print(
        "Distribución de prueba: "
        + ", ".join(
            f"{nombre}={int(cantidad)}"
            for nombre, cantidad in zip(resultado.nombres_clases, distribucion)
        )
    )
    for clasificador in resultado.clasificadores.values():
        print(f"\n--- {clasificador.nombre} ---")
        print(f"Accuracy:        {clasificador.metricas['accuracy']:.4f}")
        print(f"Precisión macro: {clasificador.metricas['precision_macro']:.4f}")
        print(f"Recall macro:    {clasificador.metricas['recall_macro']:.4f}")
        print(f"F1 macro:        {clasificador.metricas['f1_macro']:.4f}")
        print("Matriz de confusión:")
        print(clasificador.matriz_confusion)
        print("Reporte por clase:")
        print(clasificador.reporte_clasificacion)

    mejor = max(
        resultado.clasificadores.values(),
        key=lambda item: item.metricas["f1_macro"],
    )
    print(
        f"Mejor F1 macro: {mejor.nombre} "
        f"({mejor.metricas['f1_macro']:.4f})"
    )
