"""Métricas y líneas de estimación para la comparación de regresores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from housing_regression.experiment import (
    LineasEstimacion,
    ResultadoExperimentoRegresion,
    resumir_validacion,
)

COLORES_MODELO = ("#1565c0", "#ef6c00", "#2e7d32", "#7b1fa2")


def crear_figura_metricas(resultado: ResultadoExperimentoRegresion) -> Figure:
    """Compara holdout con media y desviación de validación cruzada."""
    figura, ejes = plt.subplots(1, 3, figsize=(18, 6.5))
    nombres = tuple(resultado.regresores)
    etiquetas_cortas = (
        "Lineal",
        "Polinomial",
        "Log-lineal",
        "Árbol",
    )
    posiciones = np.arange(len(nombres))
    ancho = 0.35
    configuraciones = (
        ("mae", "MAE", "Menor es mejor"),
        ("rmse", "RMSE", "Menor es mejor"),
        ("r2", "R²", "Mayor es mejor"),
    )

    for eje, (clave, titulo, criterio) in zip(ejes, configuraciones):
        prueba = [
            resultado.regresores[nombre].metricas_prueba[clave]
            for nombre in nombres
        ]
        resumen_cv = [
            resumir_validacion(resultado.regresores[nombre], clave)
            for nombre in nombres
        ]
        medias_cv = [item[0] for item in resumen_cv]
        desviaciones_cv = [item[1] for item in resumen_cv]
        barras_prueba = eje.bar(
            posiciones - ancho / 2,
            prueba,
            width=ancho,
            color="#1565c0",
            label="Prueba final (20%)",
        )
        barras_cv = eje.bar(
            posiciones + ancho / 2,
            medias_cv,
            width=ancho,
            yerr=desviaciones_cv,
            capsize=4,
            color="#90caf9",
            edgecolor="#1565c0",
            linewidth=0.8,
            label=f"CV {resultado.pliegues_cv} pliegues",
        )
        eje.bar_label(
            barras_prueba,
            labels=[f"{valor:.3f}" for valor in prueba],
            padding=3,
            fontsize=8,
        )
        eje.bar_label(
            barras_cv,
            labels=[f"{valor:.3f}" for valor in medias_cv],
            padding=3,
            fontsize=8,
        )
        eje.set_xticks(posiciones, etiquetas_cortas, rotation=15)
        eje.set_title(f"{titulo} — {criterio}")
        eje.set_ylabel("Puntuación")
        eje.grid(axis="y", alpha=0.22)
        if clave in {"mae", "rmse"}:
            eje.set_ylim(bottom=0)
        eje.legend(loc="best", fontsize=8)

    figura.suptitle(
        "California Housing — métricas de regresión",
        fontsize=17,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.015,
        "La validación cruzada se realiza únicamente sobre el 80% de entrenamiento; "
        "las barras de error muestran ±1 desviación estándar.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.055, 1, 0.94))
    return figura


def crear_figura_estimaciones(
    resultado: ResultadoExperimentoRegresion,
    lineas: LineasEstimacion,
) -> Figure:
    """Muestra curvas condicionadas y dispersión de valores reales/predichos."""
    figura = plt.figure(figsize=(16, 15))
    rejilla = figura.add_gridspec(3, 2, height_ratios=(1.1, 1.0, 1.0))
    eje_lineas = figura.add_subplot(rejilla[0, :])
    generador = np.random.default_rng(resultado.semilla)
    cantidad_muestra = min(1400, len(resultado.x_prueba))
    indices_muestra = generador.choice(
        len(resultado.x_prueba),
        size=cantidad_muestra,
        replace=False,
    )
    valores_prueba_atributo = resultado.x_prueba[:, lineas.indice_atributo]
    indices_en_rango = np.flatnonzero(
        (valores_prueba_atributo >= lineas.valores_atributo.min())
        & (valores_prueba_atributo <= lineas.valores_atributo.max())
    )
    cantidad_linea = min(1400, len(indices_en_rango))
    indices_linea = generador.choice(
        indices_en_rango,
        size=cantidad_linea,
        replace=False,
    )
    eje_lineas.scatter(
        resultado.x_prueba[indices_linea, lineas.indice_atributo],
        resultado.y_prueba[indices_linea],
        s=12,
        color="#777777",
        alpha=0.18,
        label="Observaciones de prueba",
    )
    for (nombre, predicciones), color in zip(
        lineas.predicciones.items(),
        COLORES_MODELO,
    ):
        eje_lineas.plot(
            lineas.valores_atributo,
            predicciones,
            color=color,
            linewidth=2.3,
            label=nombre,
        )
    eje_lineas.set_xlabel(lineas.nombre_atributo)
    eje_lineas.set_ylabel("Valor estimado (cientos de miles de USD)")
    eje_lineas.set_xlim(lineas.valores_atributo[[0, -1]])
    eje_lineas.set_title(
        "Líneas de estimación al variar un atributo y fijar los demás en su mediana"
    )
    eje_lineas.grid(alpha=0.22)
    eje_lineas.legend(ncol=3, fontsize=8)

    todos_predichos = np.concatenate(
        [regresor.predicciones for regresor in resultado.regresores.values()]
    )
    limite_inferior = min(0.0, float(np.quantile(todos_predichos, 0.001)))
    limite_superior = max(
        float(np.max(resultado.y_prueba)),
        float(np.quantile(todos_predichos, 0.999)),
    )
    margen = (limite_superior - limite_inferior) * 0.04
    limites = (limite_inferior - margen, limite_superior + margen)

    for indice, ((nombre, regresor), color) in enumerate(
        zip(resultado.regresores.items(), COLORES_MODELO)
    ):
        eje = figura.add_subplot(rejilla[1 + indice // 2, indice % 2])
        eje.scatter(
            resultado.y_prueba[indices_muestra],
            regresor.predicciones[indices_muestra],
            s=14,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        eje.plot(limites, limites, color="#333333", linestyle="--", linewidth=1.4)
        eje.set_xlim(limites)
        eje.set_ylim(limites)
        eje.set_aspect("equal", adjustable="box")
        eje.set_xlabel("Valor real (cientos de miles de USD)")
        eje.set_ylabel("Valor estimado (cientos de miles de USD)")
        eje.set_title(
            f"{nombre}\nR² prueba = {regresor.metricas_prueba['r2']:.3f}"
        )
        eje.grid(alpha=0.2)

    figura.suptitle(
        "California Housing — líneas y calidad de estimación",
        fontsize=17,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.018,
        "La diagonal representa una predicción perfecta. Se muestra una muestra "
        "reproducible de las observaciones para conservar legibilidad.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.045, 1, 0.96))
    return figura


def generar_visualizaciones(
    resultado: ResultadoExperimentoRegresion,
    lineas: LineasEstimacion,
    ruta_metricas: Path,
    ruta_estimaciones: Path,
    mostrar_ventana: bool,
) -> None:
    """Guarda las dos figuras y opcionalmente abre las ventanas."""
    figura_metricas = crear_figura_metricas(resultado)
    figura_estimaciones = crear_figura_estimaciones(resultado, lineas)
    ruta_metricas.parent.mkdir(parents=True, exist_ok=True)
    ruta_estimaciones.parent.mkdir(parents=True, exist_ok=True)
    figura_metricas.savefig(ruta_metricas, dpi=180, bbox_inches="tight")
    figura_estimaciones.savefig(ruta_estimaciones, dpi=180, bbox_inches="tight")
    if mostrar_ventana:
        plt.show()
    else:
        plt.close(figura_metricas)
        plt.close(figura_estimaciones)
