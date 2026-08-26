"""Pruebas del algoritmo, la política, el camino y el modo headless."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gridworld.config import Configuracion
from gridworld.environment import GridWorld
from gridworld.value_iteration import IteracionDeValores

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


class TestIteracionDeValores(unittest.TestCase):
    """Comparte un resultado convergido entre pruebas relacionadas."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Configuracion()
        cls.entorno = GridWorld(cls.config)
        cls.planificador = IteracionDeValores(cls.entorno)
        cls.resultado = cls.planificador.resolver(
            mostrar_progreso_terminal=False
        )

    def test_algoritmo_converge(self) -> None:
        self.assertTrue(self.resultado.convergio)
        self.assertLess(self.resultado.iteraciones, self.config.max_iteraciones)
        self.assertLess(
            self.resultado.historial[-1].delta,
            self.config.tolerancia,
        )

    def test_camino_empieza_y_termina_correctamente(self) -> None:
        self.assertEqual(self.resultado.camino[0], self.config.inicio)
        self.assertEqual(self.resultado.camino[-1], self.config.meta)

    def test_camino_no_cruza_obstaculos_y_es_adyacente(self) -> None:
        for estado in self.resultado.camino:
            self.assertNotIn(estado, self.config.obstaculos)
        for actual, siguiente in zip(
            self.resultado.camino,
            self.resultado.camino[1:],
        ):
            distancia = abs(actual[0] - siguiente[0]) + abs(
                actual[1] - siguiente[1]
            )
            self.assertEqual(distancia, 1)

    def test_politica_reconstruye_el_mismo_camino(self) -> None:
        reconstruido = self.planificador.reconstruir_camino(
            self.resultado.politica
        )
        self.assertEqual(reconstruido, self.resultado.camino)
        simbolos = {accion.simbolo for accion in self.resultado.politica.values()}
        self.assertTrue(simbolos.issubset({"↑", "↓", "←", "→"}))

    def test_valores_satisfacen_optimalidad_de_bellman(self) -> None:
        for estado in self.entorno.estados_validos():
            if self.entorno.es_terminal(estado):
                self.assertEqual(self.resultado.valores[estado], 0.0)
                continue
            mejor = max(
                self.entorno.transicion(estado, accion).recompensa
                + self.config.gamma
                * self.resultado.valores[
                    self.entorno.transicion(estado, accion).siguiente_estado
                ]
                for accion in self.entorno.acciones
            )
            self.assertAlmostEqual(self.resultado.valores[estado], mejor, places=6)

    def test_recompensa_acumulada_corresponde_al_camino(self) -> None:
        movimientos_normales = len(self.resultado.camino) - 2
        esperada = movimientos_normales * -1.0 + 100.0
        self.assertEqual(self.resultado.recompensa_acumulada, esperada)

    def test_modo_no_gui_crea_imagen(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            entorno_proceso = os.environ.copy()
            entorno_proceso["MPLBACKEND"] = "Agg"
            proceso = subprocess.run(
                [sys.executable, str(RAIZ_PROYECTO / "main.py"), "--no-gui"],
                cwd=temporal,
                env=entorno_proceso,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            self.assertEqual(proceso.returncode, 0, proceso.stderr)
            imagen = Path(temporal) / "output" / "resultado_final.png"
            self.assertTrue(imagen.is_file())
            self.assertGreater(imagen.stat().st_size, 10_000)
            self.assertIn("Convergió: Sí", proceso.stdout)


if __name__ == "__main__":
    unittest.main()
