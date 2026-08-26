"""Pruebas del ejercicio independiente de regresión California Housing."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from housing_regression.experiment import (
    crear_regresores,
    ejecutar_experimento_regresion,
    generar_lineas_estimacion,
    guardar_cv_detalle_csv,
    guardar_lineas_estimacion_csv,
    guardar_metricas_csv,
    guardar_predicciones_csv,
)

matplotlib.use("Agg", force=True)


class TestRegresionCaliforniaHousing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generador = np.random.default_rng(2026)
        cls.datos = generador.normal(size=(240, 8))
        ruido = generador.normal(scale=0.08, size=len(cls.datos))
        log_objetivo = (
            0.8
            + 0.28 * cls.datos[:, 0]
            - 0.14 * cls.datos[:, 1]
            + 0.07 * cls.datos[:, 2] * cls.datos[:, 3]
            + ruido
        )
        cls.objetivo = np.exp(log_objetivo)
        cls.nombres = tuple(f"Variable {indice}" for indice in range(8))
        cls.resultado = ejecutar_experimento_regresion(
            cls.datos,
            cls.objetivo,
            cls.nombres,
            pliegues_cv=3,
            profundidad_arbol=6,
            minimo_hoja=3,
        )

    def test_incluye_los_cuatro_modelos_solicitados(self) -> None:
        modelos = crear_regresores()
        self.assertEqual(
            tuple(modelos),
            (
                "Regresión lineal",
                "Regresión polinomial (grado 2)",
                "Regresión log-lineal",
                "Árbol de decisión",
            ),
        )
        polinomial = modelos["Regresión polinomial (grado 2)"]
        self.assertIsInstance(
            polinomial.named_steps["caracteristicas_polinomiales"],
            PolynomialFeatures,
        )
        self.assertIsInstance(
            modelos["Regresión log-lineal"],
            TransformedTargetRegressor,
        )
        self.assertIsInstance(modelos["Árbol de decisión"], DecisionTreeRegressor)

    def test_particion_80_20_es_reproducible(self) -> None:
        self.assertEqual(len(self.resultado.x_entrenamiento), 192)
        self.assertEqual(len(self.resultado.x_prueba), 48)
        repeticion = ejecutar_experimento_regresion(
            self.datos,
            self.objetivo,
            self.nombres,
            pliegues_cv=3,
            profundidad_arbol=6,
            minimo_hoja=3,
        )
        np.testing.assert_array_equal(
            self.resultado.indices_prueba,
            repeticion.indices_prueba,
        )

    def test_metricas_holdout_y_cross_validation_son_finitas(self) -> None:
        for regresor in self.resultado.regresores.values():
            self.assertEqual(set(regresor.metricas_prueba), {"mae", "rmse", "r2"})
            metricas_prueba = tuple(regresor.metricas_prueba.values())
            self.assertTrue(np.all(np.isfinite(metricas_prueba)))
            for metrica in ("mae", "rmse", "r2"):
                self.assertEqual(len(regresor.metricas_cv[metrica]), 3)
                self.assertTrue(np.all(np.isfinite(regresor.metricas_cv[metrica])))

    def test_lineas_varian_un_atributo_y_producen_predicciones(self) -> None:
        lineas = generar_lineas_estimacion(
            self.resultado,
            indice_atributo=0,
            cantidad_puntos=60,
        )
        self.assertEqual(len(lineas.valores_atributo), 60)
        self.assertEqual(tuple(lineas.predicciones), tuple(self.resultado.regresores))
        for predicciones in lineas.predicciones.values():
            self.assertEqual(predicciones.shape, (60,))
            self.assertTrue(np.all(np.isfinite(predicciones)))

    def test_objetivo_negativo_no_es_valido_para_log_lineal(self) -> None:
        objetivo_invalido = self.objetivo.copy()
        objetivo_invalido[0] = -1.0
        with self.assertRaisesRegex(ValueError, "no negativo"):
            ejecutar_experimento_regresion(
                self.datos,
                objetivo_invalido,
                self.nombres,
                pliegues_cv=3,
            )

    def test_exportaciones_tienen_las_filas_esperadas(self) -> None:
        lineas = generar_lineas_estimacion(
            self.resultado,
            cantidad_puntos=40,
        )
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            rutas = {
                "metricas": directorio / "metricas.csv",
                "cv": directorio / "cv.csv",
                "predicciones": directorio / "predicciones.csv",
                "lineas": directorio / "lineas.csv",
            }
            guardar_metricas_csv(self.resultado, rutas["metricas"])
            guardar_cv_detalle_csv(self.resultado, rutas["cv"])
            guardar_predicciones_csv(self.resultado, rutas["predicciones"])
            guardar_lineas_estimacion_csv(lineas, rutas["lineas"])
            esperados = {
                "metricas": 4,
                "cv": 12,
                "predicciones": 48,
                "lineas": 40,
            }
            for clave, ruta in rutas.items():
                with ruta.open(encoding="utf-8-sig", newline="") as archivo:
                    cantidad_filas = len(list(csv.DictReader(archivo)))
                    self.assertEqual(cantidad_filas, esperados[clave])

    def test_visualizaciones_generan_dos_png(self) -> None:
        from housing_regression.visualization import generar_visualizaciones

        lineas = generar_lineas_estimacion(self.resultado, cantidad_puntos=40)
        with tempfile.TemporaryDirectory() as temporal:
            ruta_metricas = Path(temporal) / "metricas.png"
            ruta_estimaciones = Path(temporal) / "estimaciones.png"
            generar_visualizaciones(
                self.resultado,
                lineas,
                ruta_metricas,
                ruta_estimaciones,
                mostrar_ventana=False,
            )
            self.assertGreater(ruta_metricas.stat().st_size, 1000)
            self.assertGreater(ruta_estimaciones.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
