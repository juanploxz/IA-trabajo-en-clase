"""Gráficas para comparar particiones de entrenamiento y prueba en Iris."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from iris_classifier.experiment import entrenar_modelos_frontera
from iris_classifier.split_comparison import ResultadoComparacionParticiones

COLORES_MODELO = ("#1565c0", "#ef6c00", "#7b1fa2")
COLORES_CLASE = ("#1976d2", "#ef6c00", "#7b1fa2")
COLORES_FONDO = ("#dceeff", "#ffedcf", "#eadcf5")


def crear_figura_metricas_particiones(
    comparacion: ResultadoComparacionParticiones,
) -> Figure:
    """Muestra tamaños y cuatro métricas en escalas compartidas."""
    figura = plt.figure(figsize=(15, 12))
    rejilla = figura.add_gridspec(3, 2, height_ratios=(0.82, 1.0, 1.0))
    etiquetas_particion = [item.etiqueta for item in comparacion.particiones]
    posiciones = np.arange(len(etiquetas_particion))

    eje_tamanos = figura.add_subplot(rejilla[0, :])
    cantidades_entrenamiento = np.array(
        [len(item.experimento.x_entrenamiento) for item in comparacion.particiones]
    )
    cantidades_prueba = np.array(
        [len(item.experimento.x_prueba) for item in comparacion.particiones]
    )
    barras_entrenamiento = eje_tamanos.bar(
        posiciones,
        cantidades_entrenamiento,
        width=0.55,
        color="#1565c0",
        label="Entrenamiento",
    )
    barras_prueba = eje_tamanos.bar(
        posiciones,
        cantidades_prueba,
        width=0.55,
        bottom=cantidades_entrenamiento,
        color="#90caf9",
        label="Prueba",
    )
    eje_tamanos.bar_label(barras_entrenamiento, label_type="center", color="white")
    eje_tamanos.bar_label(barras_prueba, label_type="center", color="#202124")
    eje_tamanos.set_xticks(posiciones, etiquetas_particion)
    eje_tamanos.set_ylabel("Número de flores")
    eje_tamanos.set_xlabel("Partición entrenamiento/prueba")
    eje_tamanos.set_title("Cantidad de datos disponible para aprender y evaluar")
    eje_tamanos.set_ylim(0, 165)
    eje_tamanos.grid(axis="y", alpha=0.2)
    eje_tamanos.legend(ncol=2, loc="upper right")

    metricas = (
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precisión macro"),
        ("recall_macro", "Recall macro"),
        ("f1_macro", "F1 macro"),
    )
    todos_valores = [
        clasificador.metricas[clave]
        for particion in comparacion.particiones
        for clasificador in particion.experimento.clasificadores.values()
        for clave, _ in metricas
    ]
    minimo_y = max(0.0, np.floor((min(todos_valores) - 0.025) * 20) / 20)
    desplazamientos = ((0, -13), (-10, -10), (10, 10))

    for indice_metrica, (clave, titulo) in enumerate(metricas):
        eje = figura.add_subplot(rejilla[1 + indice_metrica // 2, indice_metrica % 2])
        for indice_modelo, (nombre, color) in enumerate(
            zip(comparacion.nombres_clasificadores, COLORES_MODELO)
        ):
            valores = [
                item.experimento.clasificadores[nombre].metricas[clave]
                for item in comparacion.particiones
            ]
            eje.plot(
                posiciones,
                valores,
                color=color,
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=nombre,
            )
            for x, valor in zip(posiciones, valores):
                eje.annotate(
                    f"{valor:.1%}",
                    (x, valor),
                    xytext=desplazamientos[indice_modelo],
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                )
        eje.set_xticks(posiciones, etiquetas_particion)
        eje.set_ylim(minimo_y, 1.015)
        eje.set_xlabel("Partición entrenamiento/prueba")
        eje.set_ylabel("Puntuación")
        eje.set_title(titulo)
        eje.grid(alpha=0.22)

    manejadores = [
        Line2D([0], [0], color=color, marker="o", linewidth=2, label=nombre)
        for nombre, color in zip(comparacion.nombres_clasificadores, COLORES_MODELO)
    ]
    figura.legend(
        handles=manejadores,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figura.suptitle(
        "Iris — efecto de cambiar la partición",
        fontsize=17,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.035,
        "Métricas calculadas con los cuatro atributos y la misma semilla (42).",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.065, 1, 0.96))
    return figura


def crear_figura_fronteras_particiones(
    comparacion: ResultadoComparacionParticiones,
    atributos: tuple[int, int],
) -> Figure:
    """Dibuja las tres particiones por los tres clasificadores en una matriz."""
    primer_experimento = comparacion.particiones[0].experimento
    if len(atributos) != 2 or atributos[0] == atributos[1]:
        raise ValueError("Las fronteras requieren dos atributos diferentes.")
    indices_invalidos = (
        indice not in range(primer_experimento.datos.shape[1])
        for indice in atributos
    )
    if any(indices_invalidos):
        raise ValueError("El índice de atributo debe estar entre 0 y 3.")

    figura, ejes = plt.subplots(
        len(comparacion.particiones),
        len(comparacion.nombres_clasificadores),
        figsize=(17, 13),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_total = primer_experimento.datos[:, atributos]
    margen_x = np.ptp(x_total[:, 0]) * 0.12
    margen_y = np.ptp(x_total[:, 1]) * 0.12
    minimo_x = x_total[:, 0].min() - margen_x
    maximo_x = x_total[:, 0].max() + margen_x
    minimo_y = x_total[:, 1].min() - margen_y
    maximo_y = x_total[:, 1].max() + margen_y
    malla_x, malla_y = np.meshgrid(
        np.linspace(minimo_x, maximo_x, 300),
        np.linspace(minimo_y, maximo_y, 300),
    )
    puntos_malla = np.column_stack((malla_x.ravel(), malla_y.ravel()))
    mapa_fondo = ListedColormap(COLORES_FONDO)

    for fila, particion in enumerate(comparacion.particiones):
        experimento = particion.experimento
        fronteras = entrenar_modelos_frontera(experimento, atributos)
        for columna, frontera in enumerate(fronteras.values()):
            eje = ejes[fila, columna]
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
                mascara_entrenamiento = experimento.y_entrenamiento == clase
                eje.scatter(
                    experimento.x_entrenamiento[
                        mascara_entrenamiento, atributos[0]
                    ],
                    experimento.x_entrenamiento[
                        mascara_entrenamiento, atributos[1]
                    ],
                    color=color,
                    marker="o",
                    s=24,
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.78,
                )
                mascara_prueba = experimento.y_prueba == clase
                eje.scatter(
                    experimento.x_prueba[mascara_prueba, atributos[0]],
                    experimento.x_prueba[mascara_prueba, atributos[1]],
                    color=color,
                    marker="X",
                    s=50,
                    edgecolor="#202124",
                    linewidth=0.55,
                )
            eje.set_title(
                f"{frontera.nombre}\nExactitud 2D: {frontera.exactitud_2d:.2%}",
                fontsize=10,
            )
            eje.grid(alpha=0.16)
            if fila == len(comparacion.particiones) - 1:
                eje.set_xlabel(experimento.nombres_atributos[atributos[0]])
            if columna == 0:
                eje.set_ylabel(
                    f"Partición {particion.etiqueta}\n"
                    f"{experimento.nombres_atributos[atributos[1]]}"
                )

    leyenda = [
        Patch(facecolor=color, label=nombre)
        for color, nombre in zip(COLORES_CLASE, primer_experimento.nombres_clases)
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
                label="Entrenamiento",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="#202124",
                markersize=8,
                label="Prueba",
            ),
        )
    )
    figura.legend(handles=leyenda, loc="lower center", ncol=5, frameon=False)
    figura.suptitle(
        "Iris — fronteras para particiones 60/40, 70/30 y 80/20",
        fontsize=17,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.035,
        "Estas fronteras usan solo los dos atributos visibles; "
        "las métricas comparativas usan los cuatro.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.07, 1, 0.96))
    return figura


def generar_visualizaciones_particiones(
    comparacion: ResultadoComparacionParticiones,
    atributos: tuple[int, int],
    ruta_metricas: Path,
    ruta_fronteras: Path,
    mostrar_ventana: bool,
) -> None:
    """Guarda las figuras y, cuando corresponde, abre sus ventanas."""
    figura_metricas = crear_figura_metricas_particiones(comparacion)
    figura_fronteras = crear_figura_fronteras_particiones(comparacion, atributos)
    ruta_metricas.parent.mkdir(parents=True, exist_ok=True)
    ruta_fronteras.parent.mkdir(parents=True, exist_ok=True)
    figura_metricas.savefig(ruta_metricas, dpi=180, bbox_inches="tight")
    figura_fronteras.savefig(ruta_fronteras, dpi=180, bbox_inches="tight")
    if mostrar_ventana:
        plt.show()
    else:
        plt.close(figura_metricas)
        plt.close(figura_fronteras)
