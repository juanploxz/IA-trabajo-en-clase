"""Modelo determinista del entorno GridWorld."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gridworld.config import Configuracion, Estado


class Accion(Enum):
    """Acciones disponibles, con desplazamiento y símbolo de política."""

    ARRIBA = (-1, 0, "↑", "Arriba")
    DERECHA = (0, 1, "→", "Derecha")
    ABAJO = (1, 0, "↓", "Abajo")
    IZQUIERDA = (0, -1, "←", "Izquierda")

    @property
    def desplazamiento(self) -> tuple[int, int]:
        """Cambio de fila y columna asociado a la acción."""
        return self.value[0], self.value[1]

    @property
    def simbolo(self) -> str:
        """Flecha usada al representar la política."""
        return self.value[2]

    @property
    def etiqueta(self) -> str:
        """Nombre legible de la acción."""
        return self.value[3]


ACCIONES: tuple[Accion, ...] = tuple(Accion)


@dataclass(frozen=True, slots=True)
class Transicion:
    """Resultado de aplicar una acción en un estado."""

    siguiente_estado: Estado
    recompensa: float
    terminal: bool
    valida: bool


class GridWorld:
    """Tablero determinista que expone el modelo de transición completo."""

    def __init__(self, configuracion: Configuracion | None = None) -> None:
        self.configuracion = configuracion or Configuracion()
        self._estados = tuple(
            (fila, columna)
            for fila in range(self.configuracion.filas)
            for columna in range(self.configuracion.columnas)
            if (fila, columna) not in self.configuracion.obstaculos
        )
        if not self.existe_camino():
            raise ValueError("La configuración no tiene un camino entre inicio y meta.")

    @property
    def acciones(self) -> tuple[Accion, ...]:
        """Todas las acciones del MDP, incluidas las que pueden ser inválidas."""
        return ACCIONES

    def estados_validos(self) -> tuple[Estado, ...]:
        """Retorna estados transitables en orden fila-columna."""
        return self._estados

    def esta_dentro(self, estado: Estado) -> bool:
        """Indica si una coordenada pertenece al tablero."""
        fila, columna = estado
        return (
            0 <= fila < self.configuracion.filas
            and 0 <= columna < self.configuracion.columnas
        )

    def es_valido(self, estado: Estado) -> bool:
        """Indica si la coordenada está dentro y no es un obstáculo."""
        return self.esta_dentro(estado) and estado not in self.configuracion.obstaculos

    def es_terminal(self, estado: Estado) -> bool:
        """Indica si el estado corresponde a la meta."""
        return estado == self.configuracion.meta

    def transicion(self, estado: Estado, accion: Accion) -> Transicion:
        """Aplica una acción según las reglas deterministas del entorno."""
        if not self.es_valido(estado):
            raise ValueError(f"El estado {estado} no es transitable.")
        if not isinstance(accion, Accion):
            raise TypeError("La acción debe ser un miembro de Accion.")
        if self.es_terminal(estado):
            return Transicion(estado, 0.0, True, False)

        delta_fila, delta_columna = accion.desplazamiento
        candidato = (estado[0] + delta_fila, estado[1] + delta_columna)
        if not self.es_valido(candidato):
            return Transicion(
                estado,
                self.configuracion.penalizacion_invalida,
                False,
                False,
            )
        if candidato == self.configuracion.meta:
            return Transicion(
                candidato,
                self.configuracion.recompensa_meta,
                True,
                True,
            )
        return Transicion(
            candidato,
            self.configuracion.penalizacion_movimiento,
            False,
            True,
        )

    def existe_camino(self) -> bool:
        """Comprueba mediante BFS que inicio y meta estén conectados."""
        inicio = self.configuracion.inicio
        pendientes = [inicio]
        visitados = {inicio}

        while pendientes:
            estado = pendientes.pop(0)
            if estado == self.configuracion.meta:
                return True
            for accion in self.acciones:
                delta_fila, delta_columna = accion.desplazamiento
                vecino = (
                    estado[0] + delta_fila,
                    estado[1] + delta_columna,
                )
                if self.es_valido(vecino) and vecino not in visitados:
                    visitados.add(vecino)
                    pendientes.append(vecino)
        return False
