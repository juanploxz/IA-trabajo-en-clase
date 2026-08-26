"""Pruebas del ejercicio independiente de clasificación Iris."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from iris_classifier.experiment import (
    crear_clasificadores,
    ejecutar_experimento,
    entrenar_modelos_frontera,
)

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


class TestClasificacionIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resultado = ejecutar_experimento()

    def test_particion_es_70_30_y_estratificada(self) -> None:
        self.assertEqual(len(self.resultado.x_entrenamiento), 105)
        self.assertEqual(len(self.resultado.x_prueba), 45)
        np.testing.assert_array_equal(
            np.bincount(self.resultado.y_entrenamiento),
            np.array([35, 35, 35]),
        )
        np.testing.assert_array_equal(
            np.bincount(self.resultado.y_prueba),
            np.array([15, 15, 15]),
        )

    def test_incluye_los_tres_clasificadores_solicitados(self) -> None:
        self.assertEqual(
            tuple(self.resultado.clasificadores),
            ("LDA", "K-NN (k=3)", "Árbol de decisión"),
        )
        modelos = crear_clasificadores()
        knn = modelos["K-NN (k=3)"].named_steps["clasificador"]
        self.assertIsInstance(knn, KNeighborsClassifier)
        self.assertEqual(knn.n_neighbors, 3)

    def test_metricas_son_validas_y_modelos_clasifican_bien(self) -> None:
        for clasificador in self.resultado.clasificadores.values():
            self.assertEqual(clasificador.matriz_confusion.shape, (3, 3))
            self.assertEqual(int(clasificador.matriz_confusion.sum()), 45)
            for valor in clasificador.metricas.values():
                self.assertGreaterEqual(valor, 0.0)
                self.assertLessEqual(valor, 1.0)
            self.assertGreaterEqual(clasificador.metricas["accuracy"], 0.85)

    def test_resultado_es_reproducible_con_la_misma_semilla(self) -> None:
        repeticion = ejecutar_experimento(semilla=42)
        np.testing.assert_array_equal(
            self.resultado.indices_prueba,
            repeticion.indices_prueba,
        )
        for nombre in self.resultado.clasificadores:
            np.testing.assert_array_equal(
                self.resultado.clasificadores[nombre].predicciones,
                repeticion.clasificadores[nombre].predicciones,
            )

    def test_fronteras_se_ajustan_con_dos_atributos(self) -> None:
        fronteras = entrenar_modelos_frontera(self.resultado, (2, 3))
        self.assertEqual(tuple(fronteras), tuple(self.resultado.clasificadores))
        for frontera in fronteras.values():
            self.assertGreaterEqual(frontera.exactitud_2d, 0.80)

    def test_cli_no_gui_genera_png_y_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            entorno = os.environ.copy()
            entorno["MPLBACKEND"] = "Agg"
            salida = Path(temporal) / "resultados"
            proceso = subprocess.run(
                [
                    sys.executable,
                    str(RAIZ_PROYECTO / "iris_classification.py"),
                    "--no-gui",
                    "--output-dir",
                    str(salida),
                ],
                cwd=temporal,
                env=entorno,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            self.assertEqual(proceso.returncode, 0, proceso.stderr)
            esperados = (
                "iris_fronteras_decision.png",
                "iris_metricas.png",
                "iris_metricas.csv",
                "iris_predicciones.csv",
            )
            for nombre in esperados:
                ruta = salida / nombre
                self.assertTrue(ruta.is_file(), nombre)
                self.assertGreater(ruta.stat().st_size, 100)
            self.assertIn("Entrenamiento: 105 (70%)", proceso.stdout)


if __name__ == "__main__":
    unittest.main()
