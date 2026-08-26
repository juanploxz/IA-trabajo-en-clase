"""Fronteras de decisión y panel de métricas para Iris."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from iris_classifier.experiment import (
    ResultadoExperimento,
    entrenar_modelos_frontera,
)

COLORES_CLASE = ("#1976d2", "#ef6c00", "#7b1fa2")
COLORES_FONDO = ("#dceeff", "#ffedcf", "#eadcf5")
COLORES_METRICA = ("#1565c0", "#ef6c00", "#2e7d32", "#7b1fa2")


def crear_figura_fronteras(
    resultado: ResultadoExperimento,
    atributos: tuple[int, int],
) -> plt.Figure:
    """Crea una frontera por modelo usando únicamente los dos ejes elegidos."""
    fronteras = entrenar_modelos_frontera(resultado, atributos)
    figura, ejes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    x_total = resultado.datos[:, atributos]
    margen_x = (x_total[:, 0].max() - x_total[:, 0].min()) * 0.12
    margen_y = (x_total[:, 1].max() - x_total[:, 1].min()) * 0.12
    minimo_x, maximo_x = x_total[:, 0].min() - margen_x, x_total[:, 0].max() + margen_x
    minimo_y, maximo_y = x_total[:, 1].min() - margen_y, x_total[:, 1].max() + margen_y
    malla_x, malla_y = np.meshgrid(
        np.linspace(minimo_x, maximo_x, 420),
        np.linspace(minimo_y, maximo_y, 420),
    )
    puntos_malla = np.column_stack((malla_x.ravel(), malla_y.ravel()))
    mapa_fondo = ListedColormap(COLORES_FONDO)

    for eje, frontera in zip(ejes, fronteras.values()):
        prediccion_malla = frontera.modelo.predict(puntos_malla).reshape(
            malla_x.shape
        )
        eje.contourf(
            malla_x,
            malla_y,
            prediccion_malla,
            levels=(-0.5, 0.5, 1.5, 2.5),
            cmap=mapa_fondo,
            alpha=0.82,
        )
        for clase, color in enumerate(COLORES_CLASE):
            mascara_entrenamiento = resultado.y_entrenamiento == clase
            eje.scatter(
                resultado.x_entrenamiento[mascara_entrenamiento, atributos[0]],
                resultado.x_entrenamiento[mascara_entrenamiento, atributos[1]],
                color=color,
                marker="o",
                s=38,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.82,
            )
            mascara_prueba = resultado.y_prueba == clase
            eje.scatter(
                resultado.x_prueba[mascara_prueba, atributos[0]],
                resultado.x_prueba[mascara_prueba, atributos[1]],
                color=color,
                marker="X",
                s=70,
                edgecolor="#202124",
                linewidth=0.65,
            )
        eje.set_title(
            f"{frontera.nombre}\nExactitud 2D: {frontera.exactitud_2d:.2%}"
        )
        eje.set_xlabel(resultado.nombres_atributos[atributos[0]])
        eje.grid(alpha=0.18)
    ejes[0].set_ylabel(resultado.nombres_atributos[atributos[1]])

    leyenda = [
        Patch(facecolor=color, label=nombre)
        for color, nombre in zip(COLORES_CLASE, resultado.nombres_clases)
    ]
    leyenda.extend(
        (
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="white",
                markersize=7,
                label="Entrenamiento (70%)",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="#202124",
                markersize=8,
                label="Prueba (30%)",
            ),
        )
    )
    figura.legend(
        handles=leyenda,
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    figura.suptitle(
        "Iris — fronteras de decisión con dos atributos",
        fontsize=16,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.04,
        "Los modelos de esta figura se ajustan solo con los dos atributos "
        "visibles; las métricas principales usan los cuatro atributos.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.11, 1, 0.93))
    return figura


def crear_figura_metricas(resultado: ResultadoExperimento) -> plt.Figure:
    """Compara cuatro métricas y las matrices de confusión."""
    figura = plt.figure(figsize=(16, 10.5))
    rejilla = figura.add_gridspec(2, 3, height_ratios=(1.05, 1.0))
    eje_barras = figura.add_subplot(rejilla[0, :])
    nombres = tuple(resultado.clasificadores)
    claves = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    etiquetas = ("Accuracy", "Precisión macro", "Recall macro", "F1 macro")
    posiciones = np.arange(len(nombres))
    ancho = 0.19

    for indice, (clave, etiqueta, color) in enumerate(
        zip(claves, etiquetas, COLORES_METRICA)
    ):
        valores = [
            resultado.clasificadores[nombre].metricas[clave]
            for nombre in nombres
        ]
        barras = eje_barras.bar(
            posiciones + (indice - 1.5) * ancho,
            valores,
            width=ancho,
            label=etiqueta,
            color=color,
        )
        eje_barras.bar_label(
            barras,
            labels=[f"{valor:.1%}" for valor in valores],
            padding=3,
            fontsize=8,
        )
    eje_barras.set_xticks(posiciones, nombres)
    eje_barras.set_ylim(0.75, 1.035)
    eje_barras.set_ylabel("Puntuación")
    eje_barras.set_title("Métricas sobre el conjunto de prueba (30%)")
    eje_barras.grid(axis="y", alpha=0.25)
    eje_barras.legend(ncol=4, loc="lower center")

    for columna, clasificador in enumerate(resultado.clasificadores.values()):
        eje = figura.add_subplot(rejilla[1, columna])
        matriz = clasificador.matriz_confusion
        imagen = eje.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max())
        for fila in range(matriz.shape[0]):
            for col in range(matriz.shape[1]):
                umbral = matriz.max() / 2
                eje.text(
                    col,
                    fila,
                    str(int(matriz[fila, col])),
                    ha="center",
                    va="center",
                    color="white" if matriz[fila, col] > umbral else "#202124",
                    fontweight="bold",
                )
        eje.set_xticks(range(3), resultado.nombres_clases, rotation=25)
        eje.set_yticks(range(3), resultado.nombres_clases)
        eje.set_xlabel("Clase predicha")
        if columna == 0:
            eje.set_ylabel("Clase real")
        eje.set_title(f"Matriz — {clasificador.nombre}")
        figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)

    figura.suptitle(
        "Iris — comparación de clasificadores",
        fontsize=16,
        fontweight="bold",
    )
    figura.tight_layout(rect=(0, 0, 1, 0.95))
    return figura


def generar_visualizaciones(
    resultado: ResultadoExperimento,
    atributos: tuple[int, int],
    ruta_fronteras: Path,
    ruta_metricas: Path,
    mostrar_ventana: bool,
) -> None:
    """Guarda las dos figuras y opcionalmente las presenta juntas."""
    figura_fronteras = crear_figura_fronteras(resultado, atributos)
    figura_metricas = crear_figura_metricas(resultado)
    ruta_fronteras.parent.mkdir(parents=True, exist_ok=True)
    ruta_metricas.parent.mkdir(parents=True, exist_ok=True)
    figura_fronteras.savefig(ruta_fronteras, dpi=180, bbox_inches="tight")
    figura_metricas.savefig(ruta_metricas, dpi=180, bbox_inches="tight")
    if mostrar_ventana:
        plt.show()
    else:
        plt.close(figura_fronteras)
        plt.close(figura_metricas)
