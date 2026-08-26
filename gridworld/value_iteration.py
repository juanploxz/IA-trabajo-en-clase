"""Implementación manual del algoritmo Value Iteration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from gridworld.config import Estado
from gridworld.environment import Accion, GridWorld

Valores = dict[Estado, float]
Politica = dict[Estado, Accion]


@dataclass(slots=True)
class RegistroIteracion:
    """Instantánea de una iteración para registro y visualización."""

    numero: int
    delta: float
    valores: Valores
    politica: Politica


@dataclass(slots=True)
class ResultadoValueIteration:
    """Resultado completo de la planificación por valores."""

    valores: Valores
    politica: Politica
    historial: tuple[RegistroIteracion, ...]
    iteraciones: int
    convergio: bool
    camino: tuple[Estado, ...]
    recompensa_acumulada: float
    tiempo_segundos: float


class IteracionDeValores:
    """Resuelve un GridWorld conocido con optimalidad de Bellman."""

    def __init__(
        self,
        entorno: GridWorld,
        gamma: float | None = None,
        tolerancia: float | None = None,
        max_iteraciones: int | None = None,
    ) -> None:
        config = entorno.configuracion
        self.entorno = entorno
        self.gamma = config.gamma if gamma is None else gamma
        self.tolerancia = config.tolerancia if tolerancia is None else tolerancia
        self.max_iteraciones = (
            config.max_iteraciones
            if max_iteraciones is None
            else max_iteraciones
        )
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("Gamma debe pertenecer al intervalo [0, 1).")
        if self.tolerancia <= 0.0:
            raise ValueError("La tolerancia debe ser mayor que cero.")
        if self.max_iteraciones <= 0:
            raise ValueError("El máximo de iteraciones debe ser positivo.")

    def _retorno(
        self,
        estado: Estado,
        accion: Accion,
        valores: Valores,
    ) -> float:
        transicion = self.entorno.transicion(estado, accion)
        return (
            transicion.recompensa
            + self.gamma * valores[transicion.siguiente_estado]
        )

    def _mejor_accion(
        self,
        estado: Estado,
        valores: Valores,
    ) -> tuple[Accion, float]:
        """Elige el mayor retorno; el orden de Accion resuelve empates."""
        retornos = (
            (accion, self._retorno(estado, accion, valores))
            for accion in self.entorno.acciones
        )
        return max(retornos, key=lambda elemento: elemento[1])

    def extraer_politica(self, valores: Valores) -> Politica:
        """Extrae la acción voraz de cada estado no terminal."""
        politica: Politica = {}
        for estado in self.entorno.estados_validos():
            if not self.entorno.es_terminal(estado):
                politica[estado] = self._mejor_accion(estado, valores)[0]
        return politica

    def resolver(
        self,
        mostrar_progreso_terminal: bool = True,
    ) -> ResultadoValueIteration:
        """Actualiza valores sincrónicamente hasta converger o agotar el límite."""
        inicio_tiempo = perf_counter()
        valores: Valores = {
            estado: 0.0 for estado in self.entorno.estados_validos()
        }
        historial: list[RegistroIteracion] = []
        convergio = False

        for numero in range(1, self.max_iteraciones + 1):
            valores_anteriores = valores.copy()
            valores_nuevos = valores_anteriores.copy()
            delta = 0.0

            for estado in self.entorno.estados_validos():
                if self.entorno.es_terminal(estado):
                    valores_nuevos[estado] = 0.0
                    continue
                nuevo_valor = self._mejor_accion(
                    estado,
                    valores_anteriores,
                )[1]
                valores_nuevos[estado] = nuevo_valor
                delta = max(
                    delta,
                    abs(nuevo_valor - valores_anteriores[estado]),
                )

            valores = valores_nuevos
            politica_provisional = self.extraer_politica(valores)
            historial.append(
                RegistroIteracion(
                    numero=numero,
                    delta=delta,
                    valores=valores.copy(),
                    politica=politica_provisional.copy(),
                )
            )
            if mostrar_progreso_terminal:
                print(f"Iteración {numero:03d} | delta = {delta:.8f}")
            if delta < self.tolerancia:
                convergio = True
                break

        politica = self.extraer_politica(valores)
        camino, recompensa = self._seguir_politica(politica)
        tiempo = perf_counter() - inicio_tiempo
        return ResultadoValueIteration(
            valores=valores,
            politica=politica,
            historial=tuple(historial),
            iteraciones=len(historial),
            convergio=convergio,
            camino=tuple(camino),
            recompensa_acumulada=recompensa,
            tiempo_segundos=tiempo,
        )

    def reconstruir_camino(
        self,
        politica: Politica,
        limite_seguridad: int | None = None,
    ) -> tuple[Estado, ...]:
        """Sigue la política desde el inicio y detecta ciclos o bloqueos."""
        camino, _ = self._seguir_politica(politica, limite_seguridad)
        return tuple(camino)

    def _seguir_politica(
        self,
        politica: Politica,
        limite_seguridad: int | None = None,
    ) -> tuple[list[Estado], float]:
        config = self.entorno.configuracion
        limite = limite_seguridad or config.filas * config.columnas * 4
        estado = config.inicio
        camino = [estado]
        visitados = {estado}
        recompensa_total = 0.0

        for _ in range(limite):
            if estado == config.meta:
                return camino, recompensa_total
            accion = politica.get(estado)
            if accion is None:
                raise RuntimeError(f"La política no define una acción para {estado}.")
            transicion = self.entorno.transicion(estado, accion)
            siguiente = transicion.siguiente_estado
            if not transicion.valida or siguiente == estado:
                raise RuntimeError(
                    f"La política intenta un movimiento inválido desde {estado}."
                )
            recompensa_total += transicion.recompensa
            if siguiente in visitados and siguiente != config.meta:
                raise RuntimeError(
                    "La política produjo un ciclo antes de llegar a la meta."
                )
            camino.append(siguiente)
            visitados.add(siguiente)
            estado = siguiente

        raise RuntimeError(
            "Se alcanzó el límite de seguridad sin llegar a la meta."
        )
