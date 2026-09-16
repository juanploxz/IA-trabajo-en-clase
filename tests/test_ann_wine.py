"""Validación del split, ausencia de fuga, selección por CV y exportación de la ANN."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate, train_test_split

from ann_wine import ejecutar_experimento, exportar_resultados


class TestANNWine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Ejecutar CV real una sola vez y registrar sus entradas para verificar
        # que todas las tasas comparten folds y nunca reciben datos de prueba.
        with patch("ann_wine.cross_validate", wraps=cross_validate) as validacion:
            cls.resultado = ejecutar_experimento()
            cls.llamadas_cv = validacion.call_args_list

    def test_split_estratificado_reproducible_y_folds_sin_fuga(self) -> None:
        resultado = self.resultado
        wine = load_wine()
        esperado_train, esperado_test = train_test_split(
            np.arange(178), test_size=0.30, stratify=wine.target, random_state=42,
        )
        np.testing.assert_array_equal(resultado["idx_train"], esperado_train)
        np.testing.assert_array_equal(resultado["idx_test"], esperado_test)
        self.assertEqual(resultado["x_train"].shape, (124, 13))
        self.assertEqual(resultado["x_test"].shape, (54, 13))
        self.assertFalse(set(esperado_train) & set(esperado_test))
        self.assertEqual(set(esperado_train) | set(esperado_test), set(range(178)))
        for nombre in ("y_train", "y_test"):
            np.testing.assert_allclose(
                np.bincount(resultado[nombre]) / len(resultado[nombre]),
                np.bincount(wine.target) / len(wine.target), atol=0.02,
            )
        validaciones = []
        for train, validacion in resultado["folds"]:
            self.assertFalse(set(train) & set(validacion))
            self.assertEqual(set(train) | set(validacion), set(range(124)))
            indices_originales = resultado["idx_train"][validacion]
            self.assertFalse(set(indices_originales) & set(esperado_test))
            self.assertEqual(set(resultado["y_train"][validacion]), {0, 1, 2})
            validaciones.extend(validacion)
        self.assertEqual(sorted(validaciones), list(range(124)))

    def test_escalado_solo_train_sigmoide_y_probabilidades_softmax(self) -> None:
        resultado = self.resultado
        modelo = resultado["model"]
        np.testing.assert_allclose(
            modelo.named_steps["escalador"].mean_, resultado["x_train"].mean(axis=0),
        )
        self.assertEqual(modelo.named_steps["escalador"].n_samples_seen_, 124)
        ann = modelo.named_steps["ann"]
        self.assertEqual(ann.activation, "logistic")
        self.assertEqual(ann.out_activation_, "softmax")
        self.assertEqual(ann.hidden_layer_sizes, (16,))
        self.assertEqual(ann.coefs_[0].shape, (13, 16))
        self.assertEqual(ann.coefs_[1].shape, (16, 3))
        probabilidades = resultado["probabilities"]
        self.assertEqual(probabilidades.shape, (54, 3))
        self.assertTrue(np.isfinite(probabilidades).all())
        self.assertTrue(((probabilidades >= 0) & (probabilidades <= 1)).all())
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            ann.classes_[np.argmax(probabilidades, axis=1)], resultado["predictions"],
        )
        np.testing.assert_allclose(
            resultado["model_2d"].named_steps["escalador"].mean_,
            resultado["x_train"][:, [0, 6]].mean(axis=0),
        )

    def test_learning_rate_seleccionado_por_cv_con_los_mismos_folds(self) -> None:
        resultado = self.resultado
        resumen = resultado["cv_summary"]
        self.assertEqual([fila["learning_rate"] for fila in resumen], [0.001, 0.01, 0.1])
        self.assertEqual(
            resultado["best_lr"], max(resumen, key=lambda fila: fila["f1_mean"])["learning_rate"],
        )
        self.assertEqual(
            resultado["model"].named_steps["ann"].learning_rate_init, resultado["best_lr"],
        )
        self.assertEqual(len(resultado["cv_details"]), 15)
        self.assertEqual(len(self.llamadas_cv), 3)
        for llamada in self.llamadas_cv:
            np.testing.assert_array_equal(llamada.args[1], resultado["x_train"])
            np.testing.assert_array_equal(llamada.args[2], resultado["y_train"])
            self.assertIs(llamada.kwargs["cv"], resultado["folds"])
            self.assertEqual(llamada.kwargs["scoring"]["f1"], "f1_macro")
        for resumen_tasa in resumen:
            filas = [fila for fila in resultado["cv_details"]
                     if fila["learning_rate"] == resumen_tasa["learning_rate"]]
            self.assertEqual([fila["fold"] for fila in filas], [1, 2, 3, 4, 5])
            self.assertAlmostEqual(
                resumen_tasa["f1_mean"], np.mean([fila["validation_f1_macro"] for fila in filas]),
            )
            for fila in filas:
                self.assertTrue(all(np.isfinite(valor) for valor in fila.values()))

    def test_metricas_coherentes_y_costos_finitos(self) -> None:
        resultado = self.resultado
        for metricas in resultado["metrics"].values():
            self.assertTrue(all(np.isfinite(valor) for valor in metricas.values()))
            self.assertAlmostEqual(metricas["error"], 1.0 - metricas["accuracy"])
            self.assertGreaterEqual(metricas["log_loss"], 0.0)
            for nombre in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "error"):
                self.assertGreaterEqual(metricas[nombre], 0.0)
                self.assertLessEqual(metricas[nombre], 1.0)
        self.assertEqual(resultado["confusion"].shape, (3, 3))
        self.assertEqual(resultado["confusion"].sum(), 54)
        self.assertAlmostEqual(
            np.trace(resultado["confusion"]) / 54, resultado["metrics"]["prueba"]["accuracy"],
        )
        for costos in resultado["loss_curves"].values():
            self.assertGreater(len(costos), 0)
            self.assertTrue(np.isfinite(costos).all())
            self.assertTrue((np.asarray(costos) >= 0).all())

    def test_exportacion_preserva_filas_y_predicciones(self) -> None:
        conteos = {
            "ann_wine_metricas.csv": 3,
            "ann_wine_learning_rates.csv": 3,
            "ann_wine_cv.csv": 15,
            "ann_wine_predicciones.csv": 54,
            "ann_wine_costos.csv": sum(map(len, self.resultado["loss_curves"].values())),
        }
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal) / "resultados"
            exportar_resultados(self.resultado, directorio)
            for nombre, conteo in conteos.items():
                with (directorio / nombre).open(encoding="utf-8-sig", newline="") as archivo:
                    filas = list(csv.DictReader(archivo))
                self.assertEqual(len(filas), conteo, nombre)
                if nombre == "ann_wine_predicciones.csv":
                    self.assertEqual(
                        [int(fila["indice_original"]) for fila in filas],
                        self.resultado["idx_test"].tolist(),
                    )
                    self.assertEqual(
                        [int(fila["clase_predicha"]) for fila in filas],
                        self.resultado["predictions"].tolist(),
                    )

    def test_parametros_invalidos_fallan_antes_de_entrenar(self) -> None:
        casos = [
            {"learning_rates": ()}, {"learning_rates": (0,)},
            {"learning_rates": (-0.01,)}, {"learning_rates": (float("inf"),)},
            {"learning_rates": (float("nan"),)}, {"cv_folds": 1},
            {"cv_folds": 100}, {"hidden_units": 0}, {"max_iter": 0}, {"seed": -1},
        ]
        with patch("ann_wine.crear_modelo") as constructor:
            for parametros in casos:
                with self.subTest(parametros=parametros), self.assertRaises(ValueError):
                    ejecutar_experimento(**parametros)
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
