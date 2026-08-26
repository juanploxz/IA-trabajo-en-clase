"""Pruebas del agente Q-Learning aplicado al GridWorld 15×15."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

import numpy as np

from gridworld.config import Configuracion
from gridworld.environment import Accion, GridWorld
from gridworld.q_learning import AgenteQLearning, HiperparametrosQLearning
from gridworld.value_iteration import IteracionDeValores


class TestQLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Configuracion()
        cls.entorno = GridWorld(cls.config)
        cls.parametros = HiperparametrosQLearning(
            episodios=3_000,
            alpha=0.20,
            gamma=0.95,
            epsilon_inicial=1.0,
            epsilon_minimo=0.02,
            decaimiento_epsilon=0.995,
            max_pasos=500,
            semilla=42,
        )
        cls.agente = AgenteQLearning(cls.entorno, cls.parametros)
        with redirect_stdout(io.StringIO()):
            cls.resultado = cls.agente.entrenar(
                mostrar_progreso=False,
                intervalo_reporte=100,
            )
        cls.referencia = IteracionDeValores(cls.entorno).resolver(False)

    def test_actualizacion_td_respeta_formula_q_learning(self) -> None:
        agente = AgenteQLearning(
            self.entorno,
            HiperparametrosQLearning(episodios=1, alpha=0.2, gamma=0.95),
        )
        agente.tabla_q[13, 0, 0] = 10.0
        error = agente.actualizar(
            (14, 0),
            Accion.ARRIBA,
            -1.0,
            (13, 0),
            False,
        )
        self.assertAlmostEqual(error, 8.5)
        self.assertAlmostEqual(agente.tabla_q[14, 0, 0], 1.7)

    def test_tabla_q_tiene_dimensiones_correctas(self) -> None:
        self.assertEqual(self.resultado.tabla_q.shape, (15, 15, 4))
        self.assertGreater(self.resultado.total_interacciones, 0)
        self.assertGreater(self.resultado.estados_visitados, 0)

    def test_politica_llega_a_meta_sin_obstaculos(self) -> None:
        self.assertTrue(self.resultado.politica_llega_meta)
        self.assertEqual(self.resultado.camino[0], self.config.inicio)
        self.assertEqual(self.resultado.camino[-1], self.config.meta)
        for estado in self.resultado.camino:
            self.assertNotIn(estado, self.config.obstaculos)

    def test_ruta_aprendida_es_optima_como_value_iteration(self) -> None:
        self.assertEqual(
            len(self.resultado.camino),
            len(self.referencia.camino),
        )
        self.assertEqual(
            self.resultado.recompensa_acumulada,
            self.referencia.recompensa_acumulada,
        )

    def test_entrenamiento_es_reproducible(self) -> None:
        parametros = HiperparametrosQLearning(
            episodios=100,
            max_pasos=100,
            semilla=9,
        )
        primero = AgenteQLearning(self.entorno, parametros)
        segundo = AgenteQLearning(self.entorno, parametros)
        resultado_1 = primero.entrenar(False, 50)
        resultado_2 = segundo.entrenar(False, 50)
        np.testing.assert_array_equal(
            resultado_1.tabla_q,
            resultado_2.tabla_q,
        )


if __name__ == "__main__":
    unittest.main()
