"""Pruebas del ejercicio independiente de comparación de particiones Iris."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from iris_classifier.split_comparison import (
    ejecutar_comparacion_particiones,
    guardar_metricas_particiones_csv,
    guardar_predicciones_particiones_csv,
)

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


class TestComparacionParticionesIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparacion = ejecutar_comparacion_particiones()

    def test_tamanos_60_40_70_30_y_80_20(self) -> None:
        tamanos = [
            (
                item.etiqueta,
                len(item.experimento.x_entrenamiento),
                len(item.experimento.x_prueba),
            )
            for item in self.comparacion.particiones
        ]
        self.assertEqual(
            tamanos,
            [("60/40", 90, 60), ("70/30", 105, 45), ("80/20", 120, 30)],
        )

    def test_todas_las_particiones_son_estratificadas(self) -> None:
        distribuciones_esperadas = ((30, 20), (35, 15), (40, 10))
        for particion, (por_clase_entrena, por_clase_prueba) in zip(
            self.comparacion.particiones,
            distribuciones_esperadas,
        ):
            np.testing.assert_array_equal(
                np.bincount(particion.experimento.y_entrenamiento),
                np.repeat(por_clase_entrena, 3),
            )
            np.testing.assert_array_equal(
                np.bincount(particion.experimento.y_prueba),
                np.repeat(por_clase_prueba, 3),
            )

    def test_compara_tres_modelos_y_cuatro_metricas(self) -> None:
        self.assertEqual(
            self.comparacion.nombres_clasificadores,
            ("LDA", "K-NN (k=3)", "Árbol de decisión"),
        )
        for particion in self.comparacion.particiones:
            for clasificador in particion.experimento.clasificadores.values():
                self.assertEqual(
                    set(clasificador.metricas),
                    {"accuracy", "precision_macro", "recall_macro", "f1_macro"},
                )
                self.assertGreaterEqual(clasificador.metricas["accuracy"], 0.85)

    def test_comparacion_es_reproducible(self) -> None:
        repeticion = ejecutar_comparacion_particiones(semilla=42)
        for original, repetida in zip(
            self.comparacion.particiones,
            repeticion.particiones,
        ):
            np.testing.assert_array_equal(
                original.experimento.indices_prueba,
                repetida.experimento.indices_prueba,
            )

    def test_csv_contienen_nueve_metricas_y_135_predicciones(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta_metricas = Path(temporal) / "metricas.csv"
            ruta_predicciones = Path(temporal) / "predicciones.csv"
            guardar_metricas_particiones_csv(self.comparacion, ruta_metricas)
            guardar_predicciones_particiones_csv(
                self.comparacion,
                ruta_predicciones,
            )
            with ruta_metricas.open(encoding="utf-8-sig", newline="") as archivo:
                self.assertEqual(len(list(csv.DictReader(archivo))), 9)
            with ruta_predicciones.open(
                encoding="utf-8-sig",
                newline="",
            ) as archivo:
                self.assertEqual(len(list(csv.DictReader(archivo))), 135)

    def test_cli_no_gui_genera_dos_png_y_dos_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            entorno = os.environ.copy()
            entorno["MPLBACKEND"] = "Agg"
            salida = Path(temporal) / "resultados"
            proceso = subprocess.run(
                [
                    sys.executable,
                    str(RAIZ_PROYECTO / "iris_split_comparison.py"),
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
                timeout=120,
                check=False,
            )
            self.assertEqual(proceso.returncode, 0, proceso.stderr)
            esperados = (
                "iris_particiones_metricas.png",
                "iris_particiones_fronteras.png",
                "iris_particiones_metricas.csv",
                "iris_particiones_predicciones.csv",
            )
            for nombre in esperados:
                ruta = salida / nombre
                self.assertTrue(ruta.is_file(), nombre)
                self.assertGreater(ruta.stat().st_size, 100)
            self.assertIn("60/40", proceso.stdout)
            self.assertIn("80/20", proceso.stdout)


if __name__ == "__main__":
    unittest.main()

