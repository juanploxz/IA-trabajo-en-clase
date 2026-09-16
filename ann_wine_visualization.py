"""Gráficas de la ANN Wine: aprendizaje, evaluación y frontera auxiliar 2D."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


CLASS_COLORS = ("#2166ac", "#d97706", "#6b46a3")


def _figura_resultados(resultado: dict):
    """Resume el modelo de 13 atributos y la selección por CV en train."""
    figura, ejes = plt.subplots(2, 2, figsize=(14, 10.5))
    eje_costo, eje_cv, eje_metricas, eje_confusion = ejes.ravel()
    for learning_rate, costos in resultado["loss_curves"].items():
        eje_costo.plot(
            np.arange(1, len(costos) + 1), costos,
            linewidth=1.8, label=f"LR = {learning_rate:g}",
        )
    eje_costo.set_title("Costo durante el entrenamiento")
    eje_costo.set_xlabel("Época")
    eje_costo.set_ylabel("Entropía cruzada + penalización L2")
    eje_costo.grid(alpha=0.2)
    eje_costo.legend(title="Tasa de aprendizaje", fontsize=9)

    resumen = resultado["cv_summary"]
    posiciones = np.arange(len(resumen))
    medias = np.array([fila["f1_mean"] for fila in resumen])
    dispersiones = np.array([fila["f1_std"] for fila in resumen])
    eje_cv.errorbar(
        posiciones, medias, yerr=dispersiones, fmt="o", markersize=8,
        capsize=6, color="#2166ac", elinewidth=2,
        label="F1 macro de validación: media ± DE",
    )
    indice_mejor = next(
        indice for indice, fila in enumerate(resumen)
        if fila["learning_rate"] == resultado["best_lr"]
    )
    eje_cv.scatter(
        [indice_mejor], [medias[indice_mejor]], marker="*", s=190,
        color="#d97706", edgecolors="white", zorder=4, label="LR seleccionado",
    )
    # Se conserva el intervalo completo aun si media + DE supera uno.
    amplitud = max(float(np.ptp(np.r_[medias - dispersiones, medias + dispersiones])), 0.03)
    eje_cv.set_ylim(
        float(np.min(medias - dispersiones)) - 0.3 * amplitud,
        float(np.max(medias + dispersiones)) + 0.45 * amplitud,
    )
    eje_cv.set_xticks(posiciones, [f"{fila['learning_rate']:g}" for fila in resumen])
    eje_cv.set_xlim(-0.5, len(resumen) - 0.5)
    eje_cv.set_xlabel("Learning rate (LR)")
    eje_cv.set_ylabel("F1 macro")
    eje_cv.set_title("Validación cruzada: solo datos de entrenamiento")
    eje_cv.grid(axis="y", alpha=0.2)
    eje_cv.legend(loc="best", fontsize=8)

    eje_metricas.set_title("Evaluación del modelo final — 13 atributos", pad=15)
    eje_metricas.axis("off")
    etiquetas = (
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precisión macro"),
        ("recall_macro", "Recall macro"),
        ("f1_macro", "F1 macro"),
        ("error", "Error (1 − accuracy)"),
        ("log_loss", "Log-loss sin penalización"),
    )
    tabla = eje_metricas.table(
        cellText=[
            [etiqueta, f"{resultado['metrics']['entrenamiento'][clave]:.4f}",
             f"{resultado['metrics']['prueba'][clave]:.4f}"]
            for clave, etiqueta in etiquetas
        ],
        colLabels=["Métrica", f"Train (n={len(resultado['y_train'])})",
                   f"Test (n={len(resultado['y_test'])})"],
        colWidths=[0.48, 0.26, 0.26], cellLoc="center", bbox=(0, 0.17, 1, 0.79),
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    for (fila, columna), celda in tabla.get_celld().items():
        celda.set_edgecolor("#dbe2e8")
        if fila == 0:
            celda.set_facecolor("#e7eff7")
            celda.set_text_props(weight="bold")
        elif fila % 2 == 0:
            celda.set_facecolor("#f6f8fa")
        if columna == 0 and fila > 0:
            celda.set_text_props(ha="left")
    eje_metricas.text(
        0, 0.05, "Mayor es mejor en las primeras cuatro métricas.\n"
        "Menor es mejor en error y log-loss.", transform=eje_metricas.transAxes,
        fontsize=9, color="#475569", va="center",
    )

    matriz = np.asarray(resultado["confusion"])
    eje_confusion.imshow(matriz, cmap="Blues", vmin=0)
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            eje_confusion.text(
                columna, fila, str(int(matriz[fila, columna])), ha="center", va="center",
                fontsize=14,
                color="white" if matriz[fila, columna] > matriz.max() / 2 else "#172b4d",
            )
    nombres = [f"Clase {indice}" for indice in range(matriz.shape[0])]
    eje_confusion.set_xticks(np.arange(len(nombres)), nombres)
    eje_confusion.set_yticks(np.arange(len(nombres)), nombres)
    eje_confusion.set_xlabel("Clase predicha")
    eje_confusion.set_ylabel("Clase real")
    eje_confusion.set_title("Matriz de confusión — prueba final")

    figura.suptitle(
        f"Wine · ANN con sigmoide · LR seleccionado = {resultado['best_lr']:g}",
        fontsize=17, fontweight="bold", y=0.99,
    )
    figura.text(
        0.5, 0.015,
        "Las curvas de costo corresponden al entrenamiento completo para cada LR. "
        "La CV selecciona el LR sin usar el conjunto de prueba.",
        ha="center", fontsize=9, color="#475569",
    )
    figura.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=3, w_pad=3)
    return figura


def _figura_frontera(resultado: dict):
    """Entradas reales en dos dimensiones; no es una proyección de la ANN 13D."""
    indices = (0, 6)
    x_train = np.asarray(resultado["x_train"])[:, indices]
    x_test = np.asarray(resultado["x_test"])[:, indices]
    todos = np.concatenate([x_train, x_test])
    margen = np.maximum(np.ptp(todos, axis=0) * 0.08, 0.05)
    limites_inferiores = todos.min(axis=0) - margen
    limites_superiores = todos.max(axis=0) + margen
    xx, yy = np.meshgrid(
        np.linspace(limites_inferiores[0], limites_superiores[0], 250),
        np.linspace(limites_inferiores[1], limites_superiores[1], 250),
    )
    clases = resultado["model_2d"].predict(np.column_stack([xx.ravel(), yy.ravel()]))
    figura, eje = plt.subplots(figsize=(12, 8))
    eje.contourf(
        xx, yy, clases.reshape(xx.shape), levels=(-0.5, 0.5, 1.5, 2.5),
        cmap=ListedColormap(CLASS_COLORS), alpha=0.15,
    )
    for clase, color in enumerate(CLASS_COLORS):
        for valores, etiquetas, marcador, tamano in (
            (x_train, resultado["y_train"], "o", 43),
            (x_test, resultado["y_test"], "X", 90),
        ):
            mascara = np.asarray(etiquetas) == clase
            eje.scatter(
                valores[mascara, 0], valores[mascara, 1], color=color,
                marker=marcador, s=tamano, edgecolors="white", linewidths=0.7,
                alpha=0.9, zorder=3,
            )
    eje.set_xlim(limites_inferiores[0], limites_superiores[0])
    eje.set_ylim(limites_inferiores[1], limites_superiores[1])
    eje.set_xlabel("Alcohol (atributo 0)")
    eje.set_ylabel("Flavanoids (atributo 6)")
    eje.grid(alpha=0.18)
    accuracy = resultado["metrics"]["frontera_2d"]["accuracy"]
    eje.set_title(
        f"Modelo auxiliar de 2 atributos · accuracy de prueba = {accuracy:.1%}",
        fontsize=12, pad=15,
    )
    leyenda_clases = eje.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                        markeredgecolor="white", markersize=10, label=f"Clase {clase}")
                 for clase, color in enumerate(CLASS_COLORS)],
        title="Clase real / región", loc="upper left", bbox_to_anchor=(1.025, 1),
        frameon=False,
    )
    eje.add_artist(leyenda_clases)
    eje.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#475569",
                   markersize=8, label=f"Train (n={len(x_train)})"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#475569",
                   markersize=10, label=f"Test (n={len(x_test)})"),
        ],
        title="Partición", loc="upper left", bbox_to_anchor=(1.025, 0.73), frameon=False,
    )
    figura.suptitle("Wine · frontera de decisión de una ANN con sigmoide",
                     fontsize=16, fontweight="bold", y=0.98)
    figura.text(
        0.5, 0.035,
        "Esta ANN se entrena solo con Alcohol y Flavanoids para visualizar sus regiones.\n"
        "Es un modelo distinto de la ANN principal que utiliza los 13 atributos.",
        ha="center", fontsize=10, color="#475569",
    )
    figura.subplots_adjust(left=0.09, right=0.76, bottom=0.14, top=0.88)
    return figura


def generar_visualizaciones(resultado: dict, directorio: Path, mostrar_ventana: bool) -> None:
    """Guarda las dos figuras, abre ventanas si se solicita y libera las figuras."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    figuras = []
    try:
        figuras.append((_figura_resultados(resultado), "ann_wine_resultados.png"))
        figuras.append((_figura_frontera(resultado), "ann_wine_frontera.png"))
        for figura, nombre in figuras:
            figura.savefig(directorio / nombre, dpi=160, bbox_inches="tight")
        if mostrar_ventana:
            plt.show()
    finally:
        for figura, _ in figuras:
            plt.close(figura)
