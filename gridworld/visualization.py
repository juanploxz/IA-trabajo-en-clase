"""Visualización académica y animación del GridWorld con Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from gridworld.config import Estado
from gridworld.environment import Accion, GridWorld
from gridworld.value_iteration import RegistroIteracion, ResultadoValueIteration


class VisualizadorGridWorld:
    """Dibuja valores, política, ruta y agente sobre una cuadrícula."""

    COLOR_LIBRE = "#f7f3e8"
    COLOR_OBSTACULO = "#202124"
    COLOR_INICIO = "#2eaf5d"
    COLOR_META = "#dc4c4c"
    COLOR_CAMINO = "#9ed7f5"
    COLOR_AGENTE = "#ff9f1c"

    def __init__(self, entorno: GridWorld) -> None:
        self.entorno = entorno

    def _crear_figura(self) -> tuple[Figure, Axes]:
        figura, eje = plt.subplots(figsize=(12, 10))
        figura.subplots_adjust(bottom=0.14)
        return figura, eje

    def _dibujar(
        self,
        eje: Axes,
        valores: Mapping[Estado, float],
        politica: Mapping[Estado, Accion],
        camino: Sequence[Estado] = (),
        agente: Estado | None = None,
        titulo: str = "GridWorld — Value Iteration",
        mostrar_leyenda: bool = True,
    ) -> None:
        eje.clear()
        config = self.entorno.configuracion
        camino_set = set(camino)

        for fila in range(config.filas):
            for columna in range(config.columnas):
                estado = (fila, columna)
                color = self.COLOR_LIBRE
                if estado in camino_set:
                    color = self.COLOR_CAMINO
                if estado in config.obstaculos:
                    color = self.COLOR_OBSTACULO
                elif estado == config.inicio:
                    color = self.COLOR_INICIO
                elif estado == config.meta:
                    color = self.COLOR_META
                eje.add_patch(
                    Rectangle(
                        (columna, fila),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="#697077",
                        linewidth=0.65,
                    )
                )

        if camino:
            eje.plot(
                [estado[1] + 0.5 for estado in camino],
                [estado[0] + 0.5 for estado in camino],
                color="#1479b8",
                linewidth=2.2,
                alpha=0.8,
                zorder=2,
            )

        for estado, valor in valores.items():
            fila, columna = estado
            color_texto = "white" if estado in {
                config.inicio,
                config.meta,
            } else "#263238"
            eje.text(
                columna + 0.5,
                fila + 0.27,
                f"{valor:.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color=color_texto,
                zorder=3,
            )
            accion = politica.get(estado)
            if accion is not None:
                eje.text(
                    columna + 0.5,
                    fila + 0.69,
                    accion.simbolo,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color_texto,
                    zorder=3,
                )

        for etiqueta, estado in (("S", config.inicio), ("G", config.meta)):
            eje.text(
                estado[1] + 0.16,
                estado[0] + 0.72,
                etiqueta,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=4,
            )

        if agente is not None:
            eje.scatter(
                [agente[1] + 0.5],
                [agente[0] + 0.5],
                s=185,
                color=self.COLOR_AGENTE,
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
            )

        eje.set_xlim(0, config.columnas)
        eje.set_ylim(config.filas, 0)
        eje.set_aspect("equal")
        eje.set_xticks([indice + 0.5 for indice in range(config.columnas)])
        eje.set_yticks([indice + 0.5 for indice in range(config.filas)])
        eje.set_xticklabels(range(config.columnas), fontsize=7)
        eje.set_yticklabels(range(config.filas), fontsize=7)
        eje.tick_params(length=0)
        eje.set_xlabel("Columna")
        eje.set_ylabel("Fila")
        eje.set_title(titulo, fontsize=13, pad=12)

        leyenda = [
            Patch(facecolor=self.COLOR_INICIO, label="Inicio (S)"),
            Patch(facecolor=self.COLOR_META, label="Meta (G)"),
            Patch(facecolor=self.COLOR_OBSTACULO, label="Obstáculo"),
            Patch(facecolor=self.COLOR_CAMINO, label="Camino óptimo"),
            Patch(facecolor=self.COLOR_AGENTE, label="Agente"),
        ]
        if mostrar_leyenda:
            eje.legend(
                handles=leyenda,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.075),
                ncol=5,
                frameon=False,
                fontsize=8,
            )

    def dibujar_en_eje(
        self,
        eje: Axes,
        valores: Mapping[Estado, float],
        politica: Mapping[Estado, Accion],
        camino: Sequence[Estado] = (),
        agente: Estado | None = None,
        titulo: str = "GridWorld",
        mostrar_leyenda: bool = True,
    ) -> None:
        """API pública para reutilizar el tablero en figuras comparativas."""
        self._dibujar(
            eje,
            valores,
            politica,
            camino,
            agente,
            titulo,
            mostrar_leyenda,
        )

    def visualizar(
        self,
        resultado: ResultadoValueIteration,
        ruta_salida: Path,
        velocidad: float,
        mostrar_ventana: bool,
        mostrar_iteraciones: bool,
    ) -> None:
        """Muestra las dos etapas de animación y guarda el cuadro final."""
        if velocidad <= 0.0:
            raise ValueError("La velocidad debe ser mayor que cero.")
        figura, eje = self._crear_figura()

        if mostrar_ventana and mostrar_iteraciones:
            for registro in resultado.historial:
                self._dibujar_registro(eje, registro)
                figura.canvas.draw_idle()
                plt.pause(velocidad)

        if mostrar_ventana:
            total_pasos = len(resultado.camino) - 1
            for numero, estado in enumerate(resultado.camino):
                self._dibujar(
                    eje,
                    resultado.valores,
                    resultado.politica,
                    resultado.camino,
                    agente=estado,
                    titulo=(
                        "Política y camino óptimos — "
                        f"Paso {numero}/{total_pasos}"
                    ),
                )
                figura.canvas.draw_idle()
                plt.pause(velocidad)

        self._dibujar(
            eje,
            resultado.valores,
            resultado.politica,
            resultado.camino,
            agente=resultado.camino[-1],
            titulo=(
                "Resultado final de Value Iteration — "
                f"{len(resultado.camino) - 1} pasos"
            ),
        )
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=180, bbox_inches="tight")

        if mostrar_ventana:
            plt.show()
        else:
            plt.close(figura)

    def _dibujar_registro(self, eje: Axes, registro: RegistroIteracion) -> None:
        self._dibujar(
            eje,
            registro.valores,
            registro.politica,
            titulo=(
                f"Value Iteration — Iteración {registro.numero} | "
                f"delta = {registro.delta:.6g}"
            ),
        )
