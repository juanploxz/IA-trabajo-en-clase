"""Configuración y generación reproducible de escenarios GridWorld."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import random

Estado = tuple[int, int]

# Modifique estas cinco coordenadas para cambiar el escenario predeterminado.
OBSTACULOS_PREDETERMINADOS: frozenset[Estado] = frozenset(
    {
        (12, 3),
        (10, 7),
        (7, 7),
        (4, 10),
        (2, 5),
    }
)


@dataclass(frozen=True, slots=True)
class Configuracion:
    """Parámetros inmutables compartidos por los algoritmos de GridWorld."""

    filas: int = 15
    columnas: int = 15
    inicio: Estado = (14, 0)
    meta: Estado = (0, 14)
    obstaculos: frozenset[Estado] = field(
        default_factory=lambda: OBSTACULOS_PREDETERMINADOS
    )
    recompensa_meta: float = 100.0
    penalizacion_movimiento: float = -1.0
    penalizacion_invalida: float = -5.0
    gamma: float = 0.95
    tolerancia: float = 1e-6
    max_iteraciones: int = 1000

    def __post_init__(self) -> None:
        """Valida que el escenario represente un MDP coherente."""
        if self.filas <= 0 or self.columnas <= 0:
            raise ValueError("Las dimensiones del tablero deben ser positivas.")
        if len(self.obstaculos) != 5:
            raise ValueError("El escenario debe contener exactamente 5 obstáculos.")
        if not self._posicion_dentro(self.inicio):
            raise ValueError("La posición inicial está fuera del tablero.")
        if not self._posicion_dentro(self.meta):
            raise ValueError("La meta está fuera del tablero.")
        if self.inicio == self.meta:
            raise ValueError("El inicio y la meta deben ser diferentes.")
        for obstaculo in self.obstaculos:
            if not self._posicion_dentro(obstaculo):
                raise ValueError(f"El obstáculo {obstaculo} está fuera del tablero.")
        if self.inicio in self.obstaculos or self.meta in self.obstaculos:
            raise ValueError("El inicio y la meta no pueden ser obstáculos.")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("Gamma debe pertenecer al intervalo [0, 1).")
        if self.tolerancia <= 0:
            raise ValueError("La tolerancia debe ser mayor que cero.")
        if self.max_iteraciones <= 0:
            raise ValueError("El máximo de iteraciones debe ser positivo.")

    def _posicion_dentro(self, estado: Estado) -> bool:
        fila, columna = estado
        return 0 <= fila < self.filas and 0 <= columna < self.columnas

    def con_obstaculos_aleatorios(self, semilla: int) -> Configuracion:
        """Devuelve una configuración resoluble con cinco obstáculos aleatorios.

        Se usa un generador local para no modificar el estado aleatorio global.
        Una misma semilla produce la misma configuración.
        """
        generador = random.Random(semilla)
        candidatos = [
            (fila, columna)
            for fila in range(self.filas)
            for columna in range(self.columnas)
            if (fila, columna) not in {self.inicio, self.meta}
        ]

        for _ in range(10_000):
            obstaculos = frozenset(generador.sample(candidatos, 5))
            if self._existe_camino(obstaculos):
                return replace(self, obstaculos=obstaculos)
        raise RuntimeError(
            "No fue posible generar un tablero resoluble con la semilla indicada."
        )

    def _existe_camino(self, obstaculos: frozenset[Estado]) -> bool:
        """Comprueba conectividad con búsqueda en anchura."""
        pendientes: deque[Estado] = deque([self.inicio])
        visitados = {self.inicio}
        desplazamientos = ((-1, 0), (1, 0), (0, -1), (0, 1))

        while pendientes:
            estado = pendientes.popleft()
            if estado == self.meta:
                return True
            for delta_fila, delta_columna in desplazamientos:
                vecino = (
                    estado[0] + delta_fila,
                    estado[1] + delta_columna,
                )
                if (
                    self._posicion_dentro(vecino)
                    and vecino not in obstaculos
                    and vecino not in visitados
                ):
                    visitados.add(vecino)
                    pendientes.append(vecino)
        return False
