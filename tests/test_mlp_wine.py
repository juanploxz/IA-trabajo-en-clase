"""Comparación justa entre redes Wine y reproducción desde parámetros exportados."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score

from mlp_wine_compare import ejecutar_comparacion, exportar_comparacion


class TestMLPWine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resultados = ejecutar_comparacion()
        cls.temporal = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporal.cleanup)
        cls.directorio = Path(cls.temporal.name) / "resultados"
        exportar_comparacion(cls.resultados, cls.directorio)

    def leer_csv(self, nombre: str) -> list[dict[str, str]]:
        with (self.directorio / nombre).open(encoding="utf-8-sig", newline="") as archivo:
            return list(csv.DictReader(archivo))

    def test_ambas_redes_comparten_split_estratificado_y_folds(self) -> None:
        anterior, mlp = self.resultados["anterior"], self.resultados["mlp"]
        for campo in ("idx_train", "idx_test", "x_train", "x_test", "y_train", "y_test"):
            np.testing.assert_array_equal(anterior[campo], mlp[campo])
        self.assertEqual(mlp["x_train"].shape, (124, 13))
        self.assertEqual(mlp["x_test"].shape, (54, 13))
        self.assertFalse(set(mlp["idx_train"]) & set(mlp["idx_test"]))
        self.assertEqual(set(mlp["idx_train"]) | set(mlp["idx_test"]), set(range(178)))
        proporciones = np.bincount(load_wine().target) / 178
        for campo in ("y_train", "y_test"):
            np.testing.assert_allclose(
                np.bincount(mlp[campo]) / len(mlp[campo]), proporciones, atol=0.02,
            )
        self.assertEqual(len(mlp["folds"]), 5)
        self.assertEqual(len(anterior["folds"]), 5)
        validaciones = []
        for fold_anterior, fold_mlp in zip(anterior["folds"], mlp["folds"]):
            train, validacion = fold_mlp
            np.testing.assert_array_equal(fold_anterior[0], train)
            np.testing.assert_array_equal(fold_anterior[1], validacion)
            self.assertFalse(set(train) & set(validacion))
            self.assertEqual(set(train) | set(validacion), set(range(124)))
            self.assertFalse(set(mlp["idx_train"][validacion]) & set(mlp["idx_test"]))
            self.assertEqual(set(mlp["y_train"][validacion]), {0, 1, 2})
            validaciones.extend(validacion)
        self.assertEqual(sorted(validaciones), list(range(124)))

    def test_arquitecturas_sigmoide_softmax_y_escalado_solo_train(self) -> None:
        configuraciones = {"anterior": ((16,), [(13, 16), (16, 3)], 275),
                           "mlp": ((16, 8), [(13, 16), (16, 8), (8, 3)], 387)}
        for nombre, (ocultas, formas, total) in configuraciones.items():
            resultado = self.resultados[nombre]
            ann = resultado["model"].named_steps["ann"]
            escalador = resultado["model"].named_steps["escalador"]
            self.assertEqual(ann.hidden_layer_sizes, ocultas)
            self.assertEqual(ann.activation, "logistic")
            self.assertEqual(ann.out_activation_, "softmax")
            self.assertEqual([peso.shape for peso in ann.coefs_], formas)
            self.assertEqual(sum(peso.size for peso in ann.coefs_)
                             + sum(sesgo.size for sesgo in ann.intercepts_), total)
            np.testing.assert_allclose(escalador.mean_, resultado["x_train"].mean(axis=0))
            self.assertEqual(escalador.n_samples_seen_, 124)
            np.testing.assert_allclose(
                resultado["model_2d"].named_steps["escalador"].mean_,
                resultado["x_train"][:, [0, 6]].mean(axis=0),
            )
            probabilidades = resultado["probabilities"]
            self.assertEqual(probabilidades.shape, (54, 3))
            self.assertTrue(np.isfinite(probabilidades).all())
            self.assertTrue(((probabilidades >= 0) & (probabilidades <= 1)).all())
            np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0)
            np.testing.assert_array_equal(
                ann.classes_[np.argmax(probabilidades, axis=1)], resultado["predictions"],
            )

    def test_learning_rate_seleccionado_por_f1_cv_y_desempate(self) -> None:
        for resultado in self.resultados.values():
            resumen = resultado["cv_summary"]
            self.assertEqual({fila["learning_rate"] for fila in resumen}, {0.001, 0.01, 0.1})
            # Ordenar por F1 descendente y LR ascendente comprueba el desempate.
            mejor = min(resumen, key=lambda fila: (-fila["f1_mean"], fila["learning_rate"]))
            self.assertEqual(resultado["best_lr"], mejor["learning_rate"])
            self.assertEqual(resultado["model"].named_steps["ann"].learning_rate_init,
                             resultado["best_lr"])
            self.assertEqual(len(resultado["cv_details"]), 15)
            for agregado in resumen:
                filas = [fila for fila in resultado["cv_details"]
                         if fila["learning_rate"] == agregado["learning_rate"]]
                self.assertEqual([fila["fold"] for fila in filas], [1, 2, 3, 4, 5])
                f1 = [fila["validation_f1_macro"] for fila in filas]
                self.assertAlmostEqual(agregado["f1_mean"], np.mean(f1))
                self.assertAlmostEqual(agregado["f1_std"], np.std(f1, ddof=1))
                for fila in filas:
                    self.assertTrue(all(np.isfinite(valor) for valor in fila.values()))

    def test_metricas_recalculadas_y_referencia_anterior(self) -> None:
        for resultado in self.resultados.values():
            real, pred = resultado["y_test"], resultado["predictions"]
            metricas = resultado["metrics"]["prueba"]
            np.testing.assert_array_equal(resultado["confusion"],
                                          confusion_matrix(real, pred, labels=[0, 1, 2]))
            self.assertEqual(resultado["confusion"].sum(), 54)
            esperadas = {
                "accuracy": accuracy_score(real, pred),
                "error": 1.0 - accuracy_score(real, pred),
                "precision_macro": precision_score(real, pred, average="macro", zero_division=0),
                "recall_macro": recall_score(real, pred, average="macro", zero_division=0),
                "f1_macro": f1_score(real, pred, average="macro", zero_division=0),
                "log_loss": log_loss(real, resultado["probabilities"], labels=[0, 1, 2]),
            }
            for nombre, valor in esperadas.items():
                self.assertAlmostEqual(metricas[nombre], valor)
            for conjunto in resultado["metrics"].values():
                self.assertTrue(all(np.isfinite(valor) for valor in conjunto.values()))
                self.assertAlmostEqual(conjunto["error"], 1.0 - conjunto["accuracy"])
        self.assertAlmostEqual(self.resultados["anterior"]["metrics"]["prueba"]["accuracy"],
                               53 / 54)

    def test_exportacion_de_ocho_csv_completa_y_por_modelo(self) -> None:
        conteos = {
            "mlp_wine_metricas.csv": 6, "mlp_wine_cv.csv": 30,
            "mlp_wine_learning_rates.csv": 6, "mlp_wine_confusion.csv": 18,
            "mlp_wine_predicciones.csv": 108, "mlp_wine_parametros.csv": 662,
            "mlp_wine_escalado.csv": 26,
            "mlp_wine_costos.csv": sum(
                len(curva) for resultado in self.resultados.values()
                for curva in resultado["loss_curves"].values()
            ),
        }
        self.assertEqual({ruta.name for ruta in self.directorio.glob("*.csv")}, set(conteos))
        for nombre, cantidad in conteos.items():
            filas = self.leer_csv(nombre)
            self.assertEqual(len(filas), cantidad, nombre)
            self.assertEqual({fila["modelo"] for fila in filas}, {"anterior", "mlp"})
        predicciones = self.leer_csv("mlp_wine_predicciones.csv")
        confusiones = self.leer_csv("mlp_wine_confusion.csv")
        for nombre, resultado in self.resultados.items():
            filas = [fila for fila in predicciones if fila["modelo"] == nombre]
            np.testing.assert_array_equal([int(f["indice_original"]) for f in filas],
                                          resultado["idx_test"])
            np.testing.assert_array_equal([int(f["clase_real"]) for f in filas], resultado["y_test"])
            np.testing.assert_array_equal([int(f["clase_predicha"]) for f in filas],
                                          resultado["predictions"])
            np.testing.assert_allclose(
                [[float(f[f"probabilidad_clase_{c}"]) for c in range(3)] for f in filas],
                resultado["probabilities"],
            )
            for fila in (fila for fila in confusiones if fila["modelo"] == nombre):
                self.assertEqual(int(fila["cantidad"]), resultado["confusion"][
                    int(fila["clase_real"]), int(fila["clase_predicha"]),
                ])

    def test_csv_de_pesos_y_escalado_reproduce_probabilidades(self) -> None:
        parametros = self.leer_csv("mlp_wine_parametros.csv")
        escalado = self.leer_csv("mlp_wine_escalado.csv")
        for nombre, resultado in self.resultados.items():
            filas = [fila for fila in parametros if fila["modelo"] == nombre]
            escalas = {fila["atributo"]: fila for fila in escalado if fila["modelo"] == nombre}
            origenes = list(resultado["feature_names"])
            media = np.asarray([float(escalas[atributo]["media"]) for atributo in origenes])
            escala = np.asarray([float(escalas[atributo]["escala"]) for atributo in origenes])
            activacion = (resultado["x_test"] - media) / escala
            ultima = max(int(fila["capa"]) for fila in filas)
            for capa in range(1, ultima + 1):
                conexiones = [fila for fila in filas if int(fila["capa"]) == capa]
                destinos = list(dict.fromkeys(fila["destino"] for fila in conexiones))
                pesos = np.full((len(origenes), len(destinos)), np.nan)
                sesgos = np.full(len(destinos), np.nan)
                for fila in conexiones:
                    j = destinos.index(fila["destino"])
                    if fila["tipo"] == "peso":
                        i = origenes.index(fila["origen"])
                        self.assertTrue(np.isnan(pesos[i, j]), "Peso duplicado en CSV")
                        pesos[i, j] = float(fila["valor"])
                    else:
                        self.assertEqual(fila["tipo"], "sesgo")
                        self.assertTrue(np.isnan(sesgos[j]), "Sesgo duplicado en CSV")
                        sesgos[j] = float(fila["valor"])
                self.assertTrue(np.isfinite(pesos).all(), "Faltan pesos exportados")
                self.assertTrue(np.isfinite(sesgos).all(), "Faltan sesgos exportados")
                logits = activacion @ pesos + sesgos
                if capa < ultima:
                    activacion = 1.0 / (1.0 + np.exp(-logits))
                else:
                    exponenciales = np.exp(logits - logits.max(axis=1, keepdims=True))
                    activacion = exponenciales / exponenciales.sum(axis=1, keepdims=True)
                origenes = destinos
            np.testing.assert_allclose(activacion, resultado["probabilities"], rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
