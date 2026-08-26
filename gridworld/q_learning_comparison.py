"""Gráficas para comparar Q-Learning y Value Iteration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridworld.environment import GridWorld
from gridworld.q_learning import ResultadoQLearning
from gridworld.value_iteration import ResultadoValueIteration
from gridworld.visualization import VisualizadorGridWorld


def _promedio_movil(valores: np.ndarray, ventana: int) -> tuple[np.ndarray, np.ndarray]:
    if len(valores) < ventana:
        return np.arange(1, len(valores) + 1), valores
    promedio = np.convolve(valores, np.ones(ventana) / ventana, mode="valid")
    episodios = np.arange(ventana, len(valores) + 1)
    return episodios, promedio


def crear_figura_comparativa(
    entorno: GridWorld,
    q_learning: ResultadoQLearning,
    value_iteration: ResultadoValueIteration,
    ruta_salida: Path,
    mostrar_ventana: bool,
) -> None:
    """Guarda tablero, curvas de aprendizaje y tabla de métricas."""
    figura, ejes = plt.subplots(2, 2, figsize=(17, 13))
    figura.suptitle(
        "GridWorld 15×15 — Q-Learning frente a Value Iteration",
        fontsize=17,
        fontweight="bold",
    )
    visualizador = VisualizadorGridWorld(entorno)
    visualizador.dibujar_en_eje(
        ejes[0, 0],
        q_learning.valores,
        q_learning.politica,
        q_learning.camino,
        agente=(q_learning.camino[-1] if q_learning.camino else None),
        titulo="Política voraz aprendida por Q-Learning",
        mostrar_leyenda=False,
    )

    recompensas = np.array(
        [registro.recompensa for registro in q_learning.historial],
        dtype=float,
    )
    episodios, promedio_recompensas = _promedio_movil(recompensas, 100)
    ejes[0, 1].plot(
        episodios,
        promedio_recompensas,
        color="#1769aa",
        linewidth=2,
        label="Recompensa media (100 episodios)",
    )
    ejes[0, 1].axhline(
        value_iteration.recompensa_acumulada,
        color="#2e7d32",
        linestyle="--",
        label="Recompensa de ruta óptima",
    )
    ejes[0, 1].set_title("Aprendizaje de la recompensa")
    ejes[0, 1].set_xlabel("Episodio")
    ejes[0, 1].set_ylabel("Recompensa no descontada")
    ejes[0, 1].grid(alpha=0.25)
    ejes[0, 1].legend()

    exitos = np.array(
        [registro.exito for registro in q_learning.historial],
        dtype=float,
    )
    pasos = np.array(
        [registro.pasos for registro in q_learning.historial],
        dtype=float,
    )
    episodios_exito, promedio_exito = _promedio_movil(exitos, 100)
    episodios_pasos, promedio_pasos = _promedio_movil(pasos, 100)
    eje_pasos = ejes[1, 0]
    eje_exito = eje_pasos.twinx()
    linea_pasos = eje_pasos.plot(
        episodios_pasos,
        promedio_pasos,
        color="#ef6c00",
        label="Pasos medios",
    )[0]
    linea_exito = eje_exito.plot(
        episodios_exito,
        promedio_exito * 100.0,
        color="#6a1b9a",
        label="Tasa de éxito",
    )[0]
    eje_pasos.axhline(
        len(value_iteration.camino) - 1,
        color="#2e7d32",
        linestyle="--",
        label="Pasos óptimos",
    )
    eje_pasos.set_title("Eficiencia y éxito (ventana de 100)")
    eje_pasos.set_xlabel("Episodio")
    eje_pasos.set_ylabel("Pasos", color="#ef6c00")
    eje_exito.set_ylabel("Éxito (%)", color="#6a1b9a")
    eje_exito.set_ylim(0, 105)
    eje_pasos.grid(alpha=0.25)
    lineas = [linea_pasos, linea_exito] + eje_pasos.lines[1:]
    eje_pasos.legend(lineas, [linea.get_label() for linea in lineas])

    eje_tabla = ejes[1, 1]
    eje_tabla.axis("off")
    pasos_q = len(q_learning.camino) - 1 if q_learning.camino else "No llega"
    filas = [
        ["Enfoque", "Planificación", "Aprendizaje TD"],
        ["Conoce el modelo", "Sí", "No"],
        [
            "Unidad principal",
            f"{value_iteration.iteraciones} iteraciones",
            f"{q_learning.episodios} episodios",
        ],
        ["Interacciones", "0 reales", f"{q_learning.total_interacciones}"],
        [
            "Tiempo",
            f"{value_iteration.tiempo_segundos:.4f} s",
            f"{q_learning.tiempo_segundos:.4f} s",
        ],
        ["Pasos de ruta", f"{len(value_iteration.camino) - 1}", f"{pasos_q}"],
        [
            "Recompensa ruta",
            f"{value_iteration.recompensa_acumulada:.0f}",
            f"{q_learning.recompensa_acumulada:.0f}",
        ],
        ["Éxito final", "100 %", f"{q_learning.tasa_exito_final:.1%}"],
    ]
    tabla = eje_tabla.table(
        cellText=[fila[1:] for fila in filas],
        rowLabels=[fila[0] for fila in filas],
        colLabels=["Value Iteration", "Q-Learning"],
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.05, 1.8)
    eje_tabla.set_title("Comparación experimental", pad=18)

    figura.tight_layout(rect=(0, 0, 1, 0.96))
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar_ventana:
        plt.show()
    else:
        plt.close(figura)
