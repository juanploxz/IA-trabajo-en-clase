"""Gráficas de brecha de generalización y complejidad de clasificadores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from classification_fit.experiment import (
    CurvaComplejidad,
    ResultadoComparacionAjuste,
    ResultadoParticionAjuste,
)

COLOR_ENTRENAMIENTO = "#1565c0"
COLOR_PRUEBA = "#ef6c00"
COLOR_CV = "#7b1fa2"
COLOR_REFERENCIA = "#2e7d32"


def crear_figura_metricas(comparacion: ResultadoComparacionAjuste) -> Figure:
    """Compara train, test y CV para detectar brechas de generalización."""
    figura, ejes = plt.subplots(2, 2, figsize=(17, 13), squeeze=False)
    todos_valores = [
        valor
        for particion in comparacion.particiones
        for modelo in particion.modelos.values()
        for valor in (
            modelo.metricas_entrenamiento["accuracy"],
            modelo.metricas_prueba["accuracy"],
            float(np.mean(modelo.cv_validacion_accuracy)),
        )
    ]
    minimo = max(0.0, np.floor((min(todos_valores) - 0.04) * 20.0) / 20.0)

    for eje, particion in zip(ejes.ravel(), comparacion.particiones):
        modelos = list(particion.modelos.values())
        posiciones = np.arange(len(modelos))
        entrenamiento = [
            modelo.metricas_entrenamiento["accuracy"] for modelo in modelos
        ]
        prueba = [modelo.metricas_prueba["accuracy"] for modelo in modelos]
        cv_media = [
            float(np.mean(modelo.cv_validacion_accuracy)) for modelo in modelos
        ]
        cv_desviacion = [
            float(np.std(modelo.cv_validacion_accuracy, ddof=1))
            for modelo in modelos
        ]
        for posicion, inicio, fin in zip(posiciones, entrenamiento, prueba):
            eje.plot(
                (inicio, fin),
                (posicion - 0.18, posicion),
                color="#b0b0b0",
                linewidth=1.1,
                zorder=1,
            )
        eje.scatter(
            entrenamiento,
            posiciones - 0.18,
            color=COLOR_ENTRENAMIENTO,
            marker="o",
            s=52,
            label="Entrenamiento total",
            zorder=3,
        )
        eje.scatter(
            prueba,
            posiciones,
            color=COLOR_PRUEBA,
            marker="s",
            s=48,
            label="Prueba holdout",
            zorder=3,
        )
        eje.errorbar(
            cv_media,
            posiciones + 0.18,
            xerr=cv_desviacion,
            color=COLOR_CV,
            marker="D",
            linestyle="none",
            markersize=5.5,
            capsize=3,
            label="CV validación (media ± desv.)",
            zorder=3,
        )
        eje.set_yticks(posiciones, [modelo.nombre for modelo in modelos])
        eje.invert_yaxis()
        eje.set_xlim(minimo, 1.015)
        eje.set_xlabel("Accuracy")
        eje.set_title(
            f"{particion.dataset.nombre} — {particion.etiqueta_particion}\n"
            f"n train={len(particion.x_entrenamiento)}, "
            f"n test={len(particion.x_prueba)}"
        )
        eje.grid(axis="x", alpha=0.22)

    manejadores = (
        Line2D(
            [0],
            [0],
            color=COLOR_ENTRENAMIENTO,
            marker="o",
            linestyle="none",
            label="Entrenamiento total",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_PRUEBA,
            marker="s",
            linestyle="none",
            label="Prueba holdout",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_CV,
            marker="D",
            linestyle="none",
            label="CV validación (media ± desv.)",
        ),
    )
    figura.legend(
        handles=manejadores,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
    )
    figura.suptitle(
        "Wine y Breast Cancer — brecha de ajuste de los clasificadores",
        fontsize=17,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.045,
        "Una puntuación de entrenamiento alta junto a CV/prueba menor sugiere "
        "sobreajuste; puntuaciones bajas en ambos lados sugieren subajuste.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.085, 1, 0.95))
    return figura


def _dibujar_curva(
    eje: plt.Axes,
    particion: ResultadoParticionAjuste,
    curva: CurvaComplejidad,
    k_referencia: int,
) -> None:
    posiciones = np.arange(len(curva.parametros))
    eje.plot(
        posiciones,
        curva.accuracy_entrenamiento,
        color=COLOR_ENTRENAMIENTO,
        marker="o",
        linewidth=2.0,
        label="Entrenamiento total",
    )
    eje.plot(
        posiciones,
        curva.accuracy_prueba,
        color=COLOR_PRUEBA,
        marker="s",
        linewidth=2.0,
        label="Prueba holdout",
    )
    eje.errorbar(
        posiciones,
        curva.cv_validacion_media,
        yerr=curva.cv_validacion_desviacion,
        color=COLOR_CV,
        marker="D",
        linewidth=2.0,
        capsize=3,
        label="CV validación",
    )
    if curva.familia == "K-NN" and k_referencia in curva.parametros:
        indice_referencia = curva.parametros.index(k_referencia)
        eje.axvline(
            indice_referencia,
            color=COLOR_REFERENCIA,
            linestyle="--",
            linewidth=1.4,
        )
        eje.annotate(
            f"k={k_referencia}",
            (indice_referencia, 1.005),
            xytext=(5, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color=COLOR_REFERENCIA,
            fontsize=9,
        )
    eje.set_xticks(posiciones, curva.etiquetas, rotation=30, ha="right")
    minimo = min(
        float(np.min(curva.accuracy_entrenamiento)),
        float(np.min(curva.accuracy_prueba)),
        float(np.min(curva.cv_validacion_media - curva.cv_validacion_desviacion)),
    )
    eje.set_ylim(max(0.0, np.floor((minimo - 0.04) * 20) / 20), 1.015)
    eje.set_xlabel(
        "Número de vecinos k"
        if curva.familia == "K-NN"
        else "Profundidad máxima"
    )
    eje.set_ylabel("Accuracy")
    eje.set_title(
        f"{particion.dataset.nombre} — {particion.etiqueta_particion}"
    )
    eje.grid(alpha=0.22)


def crear_figura_complejidad(
    comparacion: ResultadoComparacionAjuste,
    familia: str,
) -> Figure:
    """Dibuja la curva de complejidad de K-NN o árbol para los cuatro casos."""
    if any(familia not in particion.curvas for particion in comparacion.particiones):
        raise ValueError(f"No se calculó la curva de {familia}.")
    figura, ejes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    for eje, particion in zip(ejes.ravel(), comparacion.particiones):
        _dibujar_curva(
            eje,
            particion,
            particion.curvas[familia],
            comparacion.k_referencia,
        )
    manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
    figura.legend(
        manejadores,
        etiquetas,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
    )
    titulo = (
        "K-NN: efecto del número de vecinos"
        if familia == "K-NN"
        else "Árbol de decisión: efecto de la profundidad"
    )
    nota = (
        "En K-NN, un k pequeño es más flexible; al aumentar k la frontera se "
        "suaviza y puede aparecer subajuste."
        if familia == "K-NN"
        else "Una profundidad pequeña limita el aprendizaje; sin límite, el "
        "árbol puede memorizar el entrenamiento."
    )
    figura.suptitle(titulo, fontsize=17, fontweight="bold")
    figura.text(0.5, 0.045, nota, ha="center", fontsize=9)
    figura.tight_layout(rect=(0, 0.085, 1, 0.95))
    return figura


def generar_visualizaciones(
    comparacion: ResultadoComparacionAjuste,
    ruta_metricas: Path,
    ruta_knn: Path,
    ruta_arbol: Path,
    mostrar_ventana: bool,
) -> None:
    """Guarda las tres figuras y opcionalmente abre las ventanas."""
    figura_metricas = crear_figura_metricas(comparacion)
    figura_knn = crear_figura_complejidad(comparacion, "K-NN")
    figura_arbol = crear_figura_complejidad(comparacion, "Árbol de decisión")
    for figura, ruta in (
        (figura_metricas, ruta_metricas),
        (figura_knn, ruta_knn),
        (figura_arbol, ruta_arbol),
    ):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta, dpi=180, bbox_inches="tight")
    if mostrar_ventana:
        plt.show()
    else:
        plt.close(figura_metricas)
        plt.close(figura_knn)
        plt.close(figura_arbol)
