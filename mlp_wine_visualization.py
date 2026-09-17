"""Comparación gráfica entre una y dos capas ocultas para Wine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


CLASS_COLORS = ("#2166ac", "#d97706", "#6b46a3")
MODELOS = (("anterior", "ANN anterior"), ("mlp", "MLP nuevo"))


def _arquitectura(experimento: dict, entradas: int = 13) -> str:
    """Obtiene las capas del estimador para que los títulos reflejen el modelo."""
    capas = experimento["model"].named_steps["ann"].hidden_layer_sizes
    if isinstance(capas, int):
        capas = (capas,)
    salidas = len(experimento["model"].classes_)
    return " → ".join(str(tamano) for tamano in (entradas, *capas, salidas))


def _figura_comparacion(resultado: dict):
    figura = plt.figure(figsize=(14, 10.5))
    rejilla = figura.add_gridspec(2, 2, height_ratios=(1.0, 0.86))
    matrices = [np.asarray(resultado[clave]["confusion"]) for clave, _ in MODELOS]
    maximo = max(1, *(int(matriz.max()) for matriz in matrices))
    for indice, ((clave, nombre), matriz) in enumerate(zip(MODELOS, matrices)):
        experimento = resultado[clave]
        eje = figura.add_subplot(rejilla[0, indice])
        eje.imshow(matriz, cmap="Blues", vmin=0, vmax=maximo)
        for fila in range(matriz.shape[0]):
            for columna in range(matriz.shape[1]):
                eje.text(
                    columna, fila, str(int(matriz[fila, columna])), ha="center", va="center",
                    fontsize=16, color="white" if matriz[fila, columna] > maximo / 2 else "#172b4d",
                )
        etiquetas = [f"Clase {clase}" for clase in range(len(matriz))]
        eje.set_xticks(np.arange(len(matriz)), etiquetas)
        eje.set_yticks(np.arange(len(matriz)), etiquetas)
        eje.set_xlabel("Clase predicha")
        eje.set_ylabel("Clase real")
        eje.set_title(
            f"{nombre}: {_arquitectura(experimento)}\n"
            f"Prueba final · LR seleccionado = {experimento['best_lr']:g}", pad=12,
        )

    eje_tabla = figura.add_subplot(rejilla[1, :])
    eje_tabla.axis("off")
    etiquetas_metricas = (
        ("accuracy", "Accuracy"), ("precision_macro", "Precisión macro"),
        ("recall_macro", "Recall macro"), ("f1_macro", "F1 macro"),
        ("error", "Error (1 − accuracy)"), ("log_loss", "Log-loss sin penalización"),
    )
    filas = []
    for clave_metrica, etiqueta in etiquetas_metricas:
        filas.append([etiqueta] + [
            f"{resultado[clave]['metrics'][particion][clave_metrica]:.4f}"
            for clave, _ in MODELOS for particion in ("entrenamiento", "prueba")
        ])
    columnas = ["Métrica"] + [
        f"{nombre}\n{particion} (n={len(resultado[clave][campo])})"
        for clave, nombre in MODELOS for particion, campo in (("Train", "y_train"), ("Test", "y_test"))
    ]
    tabla = eje_tabla.table(
        cellText=filas, colLabels=columnas, colWidths=[0.30, 0.175, 0.175, 0.175, 0.175],
        cellLoc="center", bbox=(0, 0.08, 1, 0.92),
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
    figura.suptitle("Wine · efecto de añadir una capa oculta", fontsize=18, fontweight="bold", y=0.985)
    figura.text(
        0.5, 0.018,
        "Mismos datos y partición para ambas arquitecturas. Matrices con escala de color común.\n"
        "Mayor es mejor en accuracy, precisión, recall y F1; menor es mejor en error y log-loss.",
        ha="center", fontsize=10, color="#475569",
    )
    figura.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.89, hspace=0.27, wspace=0.2)
    return figura


def _figura_learning_rates(resultado: dict):
    figura, ejes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")
    extremos_cv = []
    for indice, (clave, nombre) in enumerate(MODELOS):
        experimento = resultado[clave]
        eje_costo, eje_cv = ejes[:, indice]
        for learning_rate, costos in experimento["loss_curves"].items():
            eje_costo.plot(np.arange(1, len(costos) + 1), costos, linewidth=1.8,
                           label=f"LR = {learning_rate:g}")
        eje_costo.set_title(f"{nombre}: {_arquitectura(experimento)}", fontsize=13)
        eje_costo.set_xlabel("Época")
        eje_costo.set_ylabel("Entropía cruzada + penalización L2")
        eje_costo.grid(alpha=0.2)
        eje_costo.legend(title="Costo de entrenamiento", fontsize=9)

        resumen = experimento["cv_summary"]
        medias = np.array([fila["f1_mean"] for fila in resumen])
        dispersiones = np.array([fila["f1_std"] for fila in resumen])
        posiciones = np.arange(len(resumen))
        eje_cv.errorbar(
            posiciones, medias, yerr=dispersiones, fmt="o", markersize=8,
            capsize=6, color="#2166ac", elinewidth=2, label="Media ± DE entre pliegues",
        )
        seleccionado = next(indice for indice, fila in enumerate(resumen)
                             if fila["learning_rate"] == experimento["best_lr"])
        eje_cv.scatter(
            [seleccionado], [medias[seleccionado]], marker="*", s=190,
            color="#d97706", edgecolors="white", zorder=4, label="LR seleccionado por CV",
        )
        extremos_cv.extend(medias - dispersiones)
        extremos_cv.extend(medias + dispersiones)
        eje_cv.set_xticks(posiciones, [f"{fila['learning_rate']:g}" for fila in resumen])
        eje_cv.set_xlim(-0.5, len(resumen) - 0.5)
        eje_cv.set_xlabel("Learning rate (LR)")
        eje_cv.set_ylabel("F1 macro de validación")
        eje_cv.set_title(f"CV en train · LR seleccionado = {experimento['best_lr']:g}")
        eje_cv.grid(axis="y", alpha=0.2)
        eje_cv.legend(fontsize=9, loc="best")
    # Un eje común facilita comparar; nunca recortar los intervalos a [0, 1].
    rango = max(float(np.ptp(extremos_cv)), 0.03)
    ejes[1, 0].set_ylim(min(extremos_cv) - rango * 0.3, max(extremos_cv) + rango * 0.45)
    figura.suptitle("Wine · profundidad y tasa de aprendizaje", fontsize=18, fontweight="bold", y=0.99)
    figura.text(
        0.5, 0.015,
        "Arriba: costo sobre el entrenamiento completo por LR. Abajo: CV con los mismos pliegues para ambas redes.\n"
        "Cada arquitectura selecciona su LR mediante CV; el test permanece fuera de esta selección.",
        ha="center", fontsize=10, color="#475569",
    )
    figura.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=3, w_pad=3)
    return figura


def _figura_fronteras(resultado: dict):
    figura, ejes = plt.subplots(1, 2, figsize=(14, 7.7), sharex=True, sharey=True)
    todos = np.concatenate([
        np.asarray(resultado[clave][particion])[:, (0, 6)]
        for clave, _ in MODELOS for particion in ("x_train", "x_test")
    ])
    margen = np.maximum(np.ptp(todos, axis=0) * 0.08, 0.05)
    inferior, superior = todos.min(axis=0) - margen, todos.max(axis=0) + margen
    xx, yy = np.meshgrid(np.linspace(inferior[0], superior[0], 250),
                         np.linspace(inferior[1], superior[1], 250))
    malla = np.column_stack([xx.ravel(), yy.ravel()])
    for eje, (clave, nombre) in zip(ejes, MODELOS):
        experimento = resultado[clave]
        clases = experimento["model_2d"].predict(malla).reshape(xx.shape)
        eje.contourf(xx, yy, clases, levels=(-0.5, 0.5, 1.5, 2.5),
                     cmap=ListedColormap(CLASS_COLORS), alpha=0.15)
        for clase, color in enumerate(CLASS_COLORS):
            for campo_x, campo_y, marcador, tamano in (
                ("x_train", "y_train", "o", 38), ("x_test", "y_test", "X", 78),
            ):
                valores = np.asarray(experimento[campo_x])[:, (0, 6)]
                mascara = np.asarray(experimento[campo_y]) == clase
                eje.scatter(valores[mascara, 0], valores[mascara, 1], color=color, marker=marcador,
                            s=tamano, edgecolors="white", linewidths=0.7, alpha=0.9, zorder=3)
        eje.set_xlim(inferior[0], superior[0])
        eje.set_ylim(inferior[1], superior[1])
        eje.set_xlabel("Alcohol (atributo 0)")
        eje.set_ylabel("Flavanoids (atributo 6)")
        eje.grid(alpha=0.18)
        accuracy = experimento["metrics"]["frontera_2d"]["accuracy"]
        eje.set_title(f"{nombre} auxiliar: {_arquitectura(experimento, entradas=2)}\n"
                      f"Accuracy de prueba 2D = {accuracy:.1%}", fontsize=12, pad=15)
    leyendas = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="white", markersize=9, label=f"Clase {clase}")
        for clase, color in enumerate(CLASS_COLORS)
    ]
    leyendas += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#475569", markersize=8,
               label=f"Train (n={len(resultado['anterior']['y_train'])})"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#475569", markersize=10,
               label=f"Test (n={len(resultado['anterior']['y_test'])})"),
    ]
    figura.legend(handles=leyendas, loc="lower center", bbox_to_anchor=(0.5, 0.12),
                   ncol=5, frameon=False, fontsize=10)
    figura.suptitle("Wine · fronteras de decisión con una y dos capas ocultas",
                     fontsize=17, fontweight="bold", y=0.975)
    figura.text(
        0.5, 0.035,
        "Las regiones pertenecen a modelos auxiliares entrenados solo con Alcohol y Flavanoids.\n"
        "Son modelos distintos de las redes principales de 13 atributos; comparten la partición de muestras.",
        ha="center", fontsize=10, color="#475569",
    )
    figura.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.245, wspace=0.16)
    return figura


def generar_visualizaciones(resultado: dict, directorio: Path, mostrar_ventana: bool) -> None:
    """Guarda tres figuras de comparación y cierra las ventanas al terminar."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    figuras = []
    try:
        figuras.append((_figura_comparacion(resultado), "mlp_wine_comparacion.png"))
        figuras.append((_figura_learning_rates(resultado), "mlp_wine_learning_rates.png"))
        figuras.append((_figura_fronteras(resultado), "mlp_wine_fronteras.png"))
        for figura, nombre in figuras:
            figura.savefig(directorio / nombre, dpi=160, bbox_inches="tight")
        if mostrar_ventana:
            plt.show()
    finally:
        for figura, _ in figuras:
            plt.close(figura)
