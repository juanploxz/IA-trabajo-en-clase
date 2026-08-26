"""Comparación reproducible de particiones de entrenamiento y prueba en Iris."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iris_classifier.experiment import ResultadoExperimento, ejecutar_experimento

PROPORCIONES_ENTRENAMIENTO = (0.60, 0.70, 0.80)


@dataclass(slots=True)
class ResultadoParticion:
    """Resultado completo para una proporción de entrenamiento."""

    proporcion_entrenamiento: float
    experimento: ResultadoExperimento

    @property
    def proporcion_prueba(self) -> float:
        return self.experimento.proporcion_prueba

    @property
    def etiqueta(self) -> str:
        entrenamiento = round(self.proporcion_entrenamiento * 100)
        prueba = round(self.proporcion_prueba * 100)
        return f"{entrenamiento}/{prueba}"


@dataclass(slots=True)
class ResultadoComparacionParticiones:
    """Agrupa los experimentos que comparten semilla e hiperparámetros."""

    particiones: tuple[ResultadoParticion, ...]
    semilla: int
    vecinos: int
    profundidad_arbol: int | None

    @property
    def nombres_clasificadores(self) -> tuple[str, ...]:
        return tuple(self.particiones[0].experimento.clasificadores)


def ejecutar_comparacion_particiones(
    proporciones_entrenamiento: tuple[float, ...] = PROPORCIONES_ENTRENAMIENTO,
    semilla: int = 42,
    vecinos: int = 3,
    profundidad_arbol: int | None = None,
) -> ResultadoComparacionParticiones:
    """Entrena todos los modelos en cada partición estratificada solicitada."""
    if not proporciones_entrenamiento:
        raise ValueError("Debe indicarse al menos una partición.")
    if len(set(proporciones_entrenamiento)) != len(proporciones_entrenamiento):
        raise ValueError("Las proporciones de entrenamiento no deben repetirse.")

    resultados: list[ResultadoParticion] = []
    for proporcion_entrenamiento in proporciones_entrenamiento:
        if not 0.0 < proporcion_entrenamiento < 1.0:
            raise ValueError(
                "Cada proporción de entrenamiento debe pertenecer a (0, 1)."
            )
        proporcion_prueba = round(1.0 - proporcion_entrenamiento, 10)
        experimento = ejecutar_experimento(
            proporcion_prueba=proporcion_prueba,
            semilla=semilla,
            vecinos=vecinos,
            profundidad_arbol=profundidad_arbol,
        )
        resultados.append(
            ResultadoParticion(
                proporcion_entrenamiento=proporcion_entrenamiento,
                experimento=experimento,
            )
        )

    return ResultadoComparacionParticiones(
        particiones=tuple(resultados),
        semilla=semilla,
        vecinos=vecinos,
        profundidad_arbol=profundidad_arbol,
    )


def guardar_metricas_particiones_csv(
    comparacion: ResultadoComparacionParticiones,
    ruta: Path,
) -> None:
    """Guarda métricas y celdas de confusión para las nueve combinaciones."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_clases = comparacion.particiones[0].experimento.nombres_clases
    columnas_confusion = tuple(
        f"confusion_{real.lower()}_{predicha.lower()}"
        for real in nombres_clases
        for predicha in nombres_clases
    )
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "particion",
                "porcentaje_entrenamiento",
                "porcentaje_prueba",
                "muestras_entrenamiento",
                "muestras_prueba",
                "clasificador",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "errores",
                *columnas_confusion,
            )
        )
        for particion in comparacion.particiones:
            experimento = particion.experimento
            for clasificador in experimento.clasificadores.values():
                matriz = clasificador.matriz_confusion
                escritor.writerow(
                    (
                        particion.etiqueta,
                        round(particion.proporcion_entrenamiento * 100),
                        round(particion.proporcion_prueba * 100),
                        len(experimento.x_entrenamiento),
                        len(experimento.x_prueba),
                        clasificador.nombre,
                        clasificador.metricas["accuracy"],
                        clasificador.metricas["precision_macro"],
                        clasificador.metricas["recall_macro"],
                        clasificador.metricas["f1_macro"],
                        int(np.sum(experimento.y_prueba != clasificador.predicciones)),
                        *matriz.ravel().tolist(),
                    )
                )


def guardar_predicciones_particiones_csv(
    comparacion: ResultadoComparacionParticiones,
    ruta: Path,
) -> None:
    """Guarda las predicciones de prueba de cada partición y clasificador."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    primer_experimento = comparacion.particiones[0].experimento
    nombres_modelos = comparacion.nombres_clasificadores
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "particion",
                "indice_iris",
                *primer_experimento.nombres_atributos,
                "clase_real",
                *nombres_modelos,
            )
        )
        for particion in comparacion.particiones:
            experimento = particion.experimento
            for posicion, indice_original in enumerate(experimento.indices_prueba):
                escritor.writerow(
                    (
                        particion.etiqueta,
                        int(indice_original),
                        *experimento.x_prueba[posicion].tolist(),
                        experimento.nombres_clases[int(experimento.y_prueba[posicion])],
                        *(
                            experimento.nombres_clases[
                                int(
                                    experimento.clasificadores[nombre].predicciones[
                                        posicion
                                    ]
                                )
                            ]
                            for nombre in nombres_modelos
                        ),
                    )
                )


def imprimir_resumen_particiones(
    comparacion: ResultadoComparacionParticiones,
) -> None:
    """Imprime tamaños, métricas y el mejor F1 de cada clasificador."""
    print("=== Iris: comparación de particiones ===")
    print(f"Semilla compartida: {comparacion.semilla}")
    print("Todas las particiones son estratificadas.\n")
    encabezado = (
        f"{'Partición':<10} {'Entrena':>7} {'Prueba':>7} "
        f"{'Clasificador':<20} {'Accuracy':>10} {'F1 macro':>10}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for particion in comparacion.particiones:
        experimento = particion.experimento
        for clasificador in experimento.clasificadores.values():
            print(
                f"{particion.etiqueta:<10} "
                f"{len(experimento.x_entrenamiento):>7} "
                f"{len(experimento.x_prueba):>7} "
                f"{clasificador.nombre:<20} "
                f"{clasificador.metricas['accuracy']:>9.2%} "
                f"{clasificador.metricas['f1_macro']:>9.2%}"
            )

    print("\nMejor F1 macro por clasificador:")
    for nombre in comparacion.nombres_clasificadores:
        mejor = max(
            comparacion.particiones,
            key=lambda item: item.experimento.clasificadores[nombre].metricas[
                "f1_macro"
            ],
        )
        valor = mejor.experimento.clasificadores[nombre].metricas["f1_macro"]
        print(f"- {nombre}: partición {mejor.etiqueta} ({valor:.2%})")
