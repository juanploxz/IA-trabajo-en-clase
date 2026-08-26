"""Pruebas unitarias del modelo GridWorld."""

from __future__ import annotations

import unittest

from gridworld.config import Configuracion
from gridworld.environment import Accion, GridWorld


class TestGridWorld(unittest.TestCase):
    """Valida dimensiones, posiciones, obstáculos y transiciones."""

    def setUp(self) -> None:
        self.config = Configuracion()
        self.entorno = GridWorld(self.config)

    def test_dimensiones_y_cantidad_de_obstaculos(self) -> None:
        self.assertEqual((self.config.filas, self.config.columnas), (15, 15))
        self.assertEqual(len(self.config.obstaculos), 5)

    def test_inicio_y_meta_son_validos_y_diferentes(self) -> None:
        self.assertTrue(self.entorno.es_valido(self.config.inicio))
        self.assertTrue(self.entorno.es_valido(self.config.meta))
        self.assertNotEqual(self.config.inicio, self.config.meta)
        self.assertNotIn(self.config.inicio, self.config.obstaculos)
        self.assertNotIn(self.config.meta, self.config.obstaculos)

    def test_movimiento_fuera_del_tablero_permanece_y_penaliza(self) -> None:
        resultado = self.entorno.transicion(self.config.inicio, Accion.IZQUIERDA)
        self.assertEqual(resultado.siguiente_estado, self.config.inicio)
        self.assertEqual(resultado.recompensa, -5.0)
        self.assertFalse(resultado.valida)

    def test_agente_no_atraviesa_obstaculo(self) -> None:
        resultado = self.entorno.transicion((13, 3), Accion.ARRIBA)
        self.assertEqual(resultado.siguiente_estado, (13, 3))
        self.assertEqual(resultado.recompensa, -5.0)
        self.assertFalse(resultado.valida)

    def test_llegar_a_meta_es_terminal_y_recompensa(self) -> None:
        resultado = self.entorno.transicion((0, 13), Accion.DERECHA)
        self.assertEqual(resultado.siguiente_estado, self.config.meta)
        self.assertEqual(resultado.recompensa, 100.0)
        self.assertTrue(resultado.terminal)
        self.assertTrue(resultado.valida)

    def test_obstaculos_aleatorios_son_reproducibles_y_resolubles(self) -> None:
        primera = self.config.con_obstaculos_aleatorios(2026)
        segunda = self.config.con_obstaculos_aleatorios(2026)
        self.assertEqual(primera.obstaculos, segunda.obstaculos)
        self.assertEqual(len(primera.obstaculos), 5)
        self.assertTrue(GridWorld(primera).existe_camino())


if __name__ == "__main__":
    unittest.main()
