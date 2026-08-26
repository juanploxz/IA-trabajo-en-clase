"""Implementación tabular y reproducible de Q-Learning para GridWorld."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from gridworld.config import Estado
from gridworld.environment import Accion, GridWorld

Politica = dict[Estado, Accion]


@dataclass(frozen=True, slots=True)
class HiperparametrosQLearning:
    """Parámetros de entrenamiento del agente tabular."""

    episodios: int = 5_000
    alpha: float = 0.20
    gamma: float = 0.95
    epsilon_inicial: float = 1.0
    epsilon_minimo: float = 0.02
    decaimiento_epsilon: float = 0.997
    max_pasos: int = 500
    semilla: int = 42

    def __post_init__(self) -> None:
        if self.episodios <= 0 or self.max_pasos <= 0:
            raise ValueError("Episodios y máximo de pasos deben ser positivos.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("Alpha debe pertenecer al intervalo (0, 1].")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("Gamma debe pertenecer al intervalo [0, 1).")
        if not 0.0 <= self.epsilon_minimo <= self.epsilon_inicial <= 1.0:
            raise ValueError(
                "Los valores epsilon deben cumplir 0 <= mínimo <= inicial <= 1."
            )
        if not 0.0 < self.decaimiento_epsilon <= 1.0:
            raise ValueError("El decaimiento de epsilon debe pertenecer a (0, 1].")

    def epsilon(self, episodio: int) -> float:
        """Calcula la exploración del episodio usando decaimiento exponencial."""
        return max(
            self.epsilon_minimo,
            self.epsilon_inicial * self.decaimiento_epsilon ** (episodio - 1),
        )


@dataclass(frozen=True, slots=True)
class RegistroEpisodio:
    """Métricas observadas durante un episodio de interacción."""

    episodio: int
    recompensa: float
    pasos: int
    exito: bool
    epsilon: float
    max_error_td: float


@dataclass(slots=True)
class ResultadoQLearning:
    """Tabla Q, política y métricas finales del entrenamiento."""

    tabla_q: np.ndarray
    visitas: np.ndarray
    politica: Politica
    valores: dict[Estado, float]
    historial: tuple[RegistroEpisodio, ...]
    camino: tuple[Estado, ...]
    recompensa_acumulada: float
    politica_llega_meta: bool
    episodios: int
    total_interacciones: int
    estados_visitados: int
    tasa_exito_final: float
    episodio_estable: int | None
    tiempo_segundos: float


class AgenteQLearning:
    """Aprende Q(s,a) sin consultar explícitamente el modelo completo."""

    def __init__(
        self,
        entorno: GridWorld,
        hiperparametros: HiperparametrosQLearning | None = None,
    ) -> None:
        self.entorno = entorno
        self.hiperparametros = hiperparametros or HiperparametrosQLearning(
            gamma=entorno.configuracion.gamma
        )
        config = entorno.configuracion
        self.tabla_q = np.zeros(
            (config.filas, config.columnas, len(entorno.acciones)),
            dtype=np.float64,
        )
        self.visitas = np.zeros_like(self.tabla_q, dtype=np.int64)
        self._indice_accion = {
            accion: indice for indice, accion in enumerate(entorno.acciones)
        }

    def seleccionar_accion(
        self,
        estado: Estado,
        epsilon: float,
        generador: np.random.Generator,
    ) -> Accion:
        """Aplica una conducta epsilon-greedy con desempate aleatorio."""
        if generador.random() < epsilon:
            indice = int(generador.integers(len(self.entorno.acciones)))
            return self.entorno.acciones[indice]
        fila, columna = estado
        valores = self.tabla_q[fila, columna]
        mejores = np.flatnonzero(np.isclose(valores, valores.max()))
        indice = int(generador.choice(mejores))
        return self.entorno.acciones[indice]

    def actualizar(
        self,
        estado: Estado,
        accion: Accion,
        recompensa: float,
        siguiente_estado: Estado,
        terminal: bool,
    ) -> float:
        """Aplica una actualización TD y retorna el error utilizado."""
        fila, columna = estado
        indice = self._indice_accion[accion]
        valor_actual = self.tabla_q[fila, columna, indice]
        if terminal:
            objetivo = recompensa
        else:
            siguiente_fila, siguiente_columna = siguiente_estado
            objetivo = recompensa + self.hiperparametros.gamma * float(
                self.tabla_q[siguiente_fila, siguiente_columna].max()
            )
        error_td = objetivo - valor_actual
        self.tabla_q[fila, columna, indice] += (
            self.hiperparametros.alpha * error_td
        )
        self.visitas[fila, columna, indice] += 1
        return float(error_td)

    def extraer_politica(self) -> Politica:
        """Obtiene la política voraz; el orden de acciones resuelve empates."""
        politica: Politica = {}
        for estado in self.entorno.estados_validos():
            if self.entorno.es_terminal(estado):
                continue
            fila, columna = estado
            indice = int(np.argmax(self.tabla_q[fila, columna]))
            politica[estado] = self.entorno.acciones[indice]
        return politica

    def extraer_valores(self) -> dict[Estado, float]:
        """Aproxima V(s) mediante el máximo de la tabla Q."""
        valores: dict[Estado, float] = {}
        for estado in self.entorno.estados_validos():
            fila, columna = estado
            valores[estado] = float(self.tabla_q[fila, columna].max())
        return valores

    def seguir_politica(
        self,
        politica: Politica,
        limite_seguridad: int | None = None,
    ) -> tuple[tuple[Estado, ...], float]:
        """Evalúa la política voraz y rechaza ciclos o acciones inválidas."""
        config = self.entorno.configuracion
        limite = limite_seguridad or config.filas * config.columnas * 4
        estado = config.inicio
        camino = [estado]
        visitados = {estado}
        recompensa_total = 0.0

        for _ in range(limite):
            if estado == config.meta:
                return tuple(camino), recompensa_total
            accion = politica.get(estado)
            if accion is None:
                raise RuntimeError(f"La política Q no define una acción para {estado}.")
            transicion = self.entorno.transicion(estado, accion)
            siguiente = transicion.siguiente_estado
            if not transicion.valida or siguiente == estado:
                raise RuntimeError(
                    f"La política Q intenta un movimiento inválido desde {estado}."
                )
            recompensa_total += transicion.recompensa
            if siguiente in visitados and siguiente != config.meta:
                raise RuntimeError("La política Q produjo un ciclo.")
            camino.append(siguiente)
            visitados.add(siguiente)
            estado = siguiente
        raise RuntimeError("La política Q excedió el límite de seguridad.")

    @staticmethod
    def _primer_episodio_estable(
        historial: list[RegistroEpisodio],
        ventana: int = 100,
        umbral: float = 0.95,
    ) -> int | None:
        exitos = np.fromiter(
            (registro.exito for registro in historial),
            dtype=np.float64,
        )
        if len(exitos) < ventana:
            return None
        promedios = np.convolve(
            exitos,
            np.ones(ventana) / ventana,
            mode="valid",
        )
        candidatos = np.flatnonzero(promedios >= umbral)
        if len(candidatos) == 0:
            return None
        return int(candidatos[0] + ventana)

    def entrenar(
        self,
        mostrar_progreso: bool = True,
        intervalo_reporte: int = 250,
    ) -> ResultadoQLearning:
        """Ejecuta episodios desde el inicio usando conducta epsilon-greedy."""
        if intervalo_reporte <= 0:
            raise ValueError("El intervalo de reporte debe ser positivo.")
        parametros = self.hiperparametros
        generador = np.random.default_rng(parametros.semilla)
        historial: list[RegistroEpisodio] = []
        inicio_tiempo = perf_counter()

        for episodio in range(1, parametros.episodios + 1):
            estado = self.entorno.configuracion.inicio
            epsilon = parametros.epsilon(episodio)
            recompensa_total = 0.0
            max_error_td = 0.0
            exito = False

            for paso in range(1, parametros.max_pasos + 1):
                accion = self.seleccionar_accion(estado, epsilon, generador)
                transicion = self.entorno.transicion(estado, accion)
                error_td = self.actualizar(
                    estado,
                    accion,
                    transicion.recompensa,
                    transicion.siguiente_estado,
                    transicion.terminal,
                )
                max_error_td = max(max_error_td, abs(error_td))
                recompensa_total += transicion.recompensa
                estado = transicion.siguiente_estado
                if transicion.terminal:
                    exito = True
                    break

            registro = RegistroEpisodio(
                episodio=episodio,
                recompensa=recompensa_total,
                pasos=paso,
                exito=exito,
                epsilon=epsilon,
                max_error_td=max_error_td,
            )
            historial.append(registro)
            if mostrar_progreso and (
                episodio == 1
                or episodio % intervalo_reporte == 0
                or episodio == parametros.episodios
            ):
                recientes = historial[-min(100, len(historial)) :]
                tasa = sum(item.exito for item in recientes) / len(recientes)
                promedio = sum(item.recompensa for item in recientes) / len(recientes)
                print(
                    f"Episodio {episodio:05d}/{parametros.episodios:05d} | "
                    f"epsilon={epsilon:.4f} | recompensa={promedio:7.2f} | "
                    f"éxito(100)={tasa:.1%} | max|TD|={max_error_td:.5f}"
                )

        tiempo = perf_counter() - inicio_tiempo
        politica = self.extraer_politica()
        try:
            camino, recompensa = self.seguir_politica(politica)
            llega_meta = True
        except RuntimeError:
            camino, recompensa = tuple(), float("nan")
            llega_meta = False

        ventana_final = historial[-min(100, len(historial)) :]
        tasa_final = sum(item.exito for item in ventana_final) / len(ventana_final)
        estados_visitados = int(np.sum(np.any(self.visitas > 0, axis=2)))
        return ResultadoQLearning(
            tabla_q=self.tabla_q.copy(),
            visitas=self.visitas.copy(),
            politica=politica,
            valores=self.extraer_valores(),
            historial=tuple(historial),
            camino=camino,
            recompensa_acumulada=recompensa,
            politica_llega_meta=llega_meta,
            episodios=parametros.episodios,
            total_interacciones=int(self.visitas.sum()),
            estados_visitados=estados_visitados,
            tasa_exito_final=tasa_final,
            episodio_estable=self._primer_episodio_estable(historial),
            tiempo_segundos=tiempo,
        )

    def guardar(self, ruta: Path, resultado: ResultadoQLearning) -> None:
        """Guarda tabla Q, visitas e hiperparámetros sin usar pickle."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        parametros = self.hiperparametros
        np.savez_compressed(
            ruta,
            tabla_q=resultado.tabla_q,
            visitas=resultado.visitas,
            episodios=np.array([parametros.episodios]),
            alpha=np.array([parametros.alpha]),
            gamma=np.array([parametros.gamma]),
            epsilon_inicial=np.array([parametros.epsilon_inicial]),
            epsilon_minimo=np.array([parametros.epsilon_minimo]),
            decaimiento_epsilon=np.array([parametros.decaimiento_epsilon]),
            max_pasos=np.array([parametros.max_pasos]),
            semilla=np.array([parametros.semilla]),
        )
