"""Pruebas del ejercicio de subajuste y sobreajuste."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from classification_fit.experiment import (
    cargar_datasets,
    crear_clasificadores,
    ejecutar_comparacion_ajuste,
    guardar_complejidad_csv,
    guardar_cv_csv,
    guardar_metricas_csv,
    guardar_predicciones_csv,
)

matplotlib.use("Agg", force=True)


class TestComparacionAjusteClasificacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparacion = ejecutar_comparacion_ajuste()

    def test_carga_wine_y_breast_cancer_sin_internet(self) -> None:
        datasets = cargar_datasets()
        self.assertEqual(tuple(datasets), ("wine", "breast_cancer"))
        self.assertEqual(datasets["wine"].datos.shape, (178, 13))
        self.assertEqual(datasets["breast_cancer"].datos.shape, (569, 30))
        self.assertEqual(len(np.unique(datasets["wine"].etiquetas)), 3)
        self.assertEqual(len(np.unique(datasets["breast_cancer"].etiquetas)), 2)

    def test_configuraciones_incluyen_subajuste_referencia_y_sobreajuste(self) -> None:
        modelos = crear_clasificadores()
        self.assertEqual(len(modelos), 6)
        self.assertIn("K-NN (k=1)", modelos)
        self.assertIn("K-NN (k=7)", modelos)
        self.assertIn("K-NN (k=51)", modelos)
        self.assertIn("Árbol (profundidad 1)", modelos)
        self.assertIn("Árbol (sin límite)", modelos)
        knn = modelos["K-NN (k=7)"][0]
        self.assertIsInstance(knn, Pipeline)
        self.assertIsInstance(knn.named_steps["clasificador"], KNeighborsClassifier)
        arbol = modelos["Árbol (sin límite)"][0]
        self.assertIsInstance(arbol, DecisionTreeClassifier)
        self.assertIsNone(arbol.max_depth)

    def test_particiones_50_50_y_40_60_son_estratificadas(self) -> None:
        self.assertEqual(len(self.comparacion.particiones), 4)
        for particion in self.comparacion.particiones:
            self.assertIn(particion.etiqueta_particion, {"50/50", "40/60"})
            total = len(particion.dataset.datos)
            esperado_entrenamiento = int(
                total * particion.proporcion_entrenamiento
            )
            self.assertEqual(len(particion.x_entrenamiento), esperado_entrenamiento)
            self.assertEqual(
                len(particion.x_entrenamiento) + len(particion.x_prueba),
                total,
            )
            proporcion_total = np.bincount(particion.dataset.etiquetas) / total
            proporcion_train = np.bincount(particion.y_entrenamiento) / len(
                particion.y_entrenamiento
            )
            np.testing.assert_allclose(
                proporcion_train,
                proporcion_total,
                atol=0.02,
            )

    def test_metricas_y_validacion_cruzada_son_finitas(self) -> None:
        for particion in self.comparacion.particiones:
            for modelo in particion.modelos.values():
                for metricas in (
                    modelo.metricas_entrenamiento,
                    modelo.metricas_prueba,
                ):
                    self.assertEqual(
                        set(metricas),
                        {
                            "accuracy",
                            "balanced_accuracy",
                            "precision_macro",
                            "recall_macro",
                            "f1_macro",
                        },
                    )
                    self.assertTrue(
                        all(0.0 <= valor <= 1.0 for valor in metricas.values())
                    )
                self.assertEqual(
                    len(modelo.cv_validacion_accuracy),
                    self.comparacion.pliegues_cv,
                )
                self.assertTrue(np.all(np.isfinite(modelo.cv_validacion_accuracy)))

    def test_diagnostico_encuentra_ambos_patrones_en_cada_dataset(self) -> None:
        for clave in ("wine", "breast_cancer"):
            diagnosticos = {
                modelo.diagnostico_observado
                for particion in self.comparacion.particiones
                if particion.dataset.clave == clave
                for modelo in particion.modelos.values()
            }
            self.assertIn("Subajuste observado", diagnosticos)
            self.assertIn("Sobreajuste observado", diagnosticos)

    def test_curvas_incluyen_k7_y_arbol_sin_limite(self) -> None:
        for particion in self.comparacion.particiones:
            curva_knn = particion.curvas["K-NN"]
            curva_arbol = particion.curvas["Árbol de decisión"]
            self.assertIn(7, curva_knn.parametros)
            self.assertIn(51, curva_knn.parametros)
            self.assertIn(None, curva_arbol.parametros)
            self.assertEqual(
                len(curva_knn.parametros),
                len(curva_knn.cv_validacion_media),
            )

    def test_exportaciones_tienen_las_filas_esperadas(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            rutas = {
                "metricas": directorio / "metricas.csv",
                "cv": directorio / "cv.csv",
                "complejidad": directorio / "complejidad.csv",
                "predicciones": directorio / "predicciones.csv",
            }
            guardar_metricas_csv(self.comparacion, rutas["metricas"])
            guardar_cv_csv(self.comparacion, rutas["cv"])
            guardar_complejidad_csv(self.comparacion, rutas["complejidad"])
            guardar_predicciones_csv(self.comparacion, rutas["predicciones"])
            cantidad_modelos = sum(
                len(particion.modelos)
                for particion in self.comparacion.particiones
            )
            esperados = {
                "metricas": cantidad_modelos,
                "cv": cantidad_modelos * self.comparacion.pliegues_cv,
                "complejidad": sum(
                    len(curva.parametros)
                    for particion in self.comparacion.particiones
                    for curva in particion.curvas.values()
                ),
                "predicciones": sum(
                    len(particion.y_prueba)
                    for particion in self.comparacion.particiones
                ),
            }
            for clave, ruta in rutas.items():
                with ruta.open(encoding="utf-8-sig", newline="") as archivo:
                    self.assertEqual(len(list(csv.DictReader(archivo))), esperados[clave])

    def test_visualizaciones_generan_tres_png(self) -> None:
        from classification_fit.visualization import generar_visualizaciones

        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            rutas = (
                directorio / "metricas.png",
                directorio / "knn.png",
                directorio / "arbol.png",
            )
            generar_visualizaciones(
                self.comparacion,
                *rutas,
                mostrar_ventana=False,
            )
            for ruta in rutas:
                self.assertGreater(ruta.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
