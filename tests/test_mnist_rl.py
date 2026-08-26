"""Pruebas locales de la política RL y utilidades de imágenes MNIST."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from mnist_rl import (
    PoliticaMNISTRL,
    construir_mosaico,
    normalizar_imagen_externa,
    seleccionar_digito,
)


class TestMNISTRL(unittest.TestCase):
    def test_politica_produce_distribuciones_validas(self) -> None:
        modelo = PoliticaMNISTRL.crear(7)
        entradas = np.zeros((4, 784), dtype=np.float32)
        probabilidades = modelo.probabilidades(entradas)
        self.assertEqual(probabilidades.shape, (4, 10))
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0)
        self.assertTrue(np.all(probabilidades >= 0.0))

    def test_entrenamiento_reinforce_actualiza_pesos(self) -> None:
        generador = np.random.default_rng(3)
        imagenes = generador.integers(
            0,
            256,
            size=(40, 28, 28),
            dtype=np.uint8,
        )
        etiquetas = generador.integers(0, 10, size=40, dtype=np.uint8)
        modelo = PoliticaMNISTRL.crear(3)
        originales = modelo.pesos.copy()
        with redirect_stdout(io.StringIO()):
            historial = modelo.entrenar(
                imagenes,
                etiquetas,
                imagenes,
                etiquetas,
                epocas=1,
                tamano_lote=20,
                tasa_aprendizaje=0.005,
                semilla=3,
            )
        self.assertFalse(np.array_equal(originales, modelo.pesos))
        self.assertEqual(len(historial), 1)
        self.assertIn("exactitud_prueba", historial[0])

    def test_modelo_se_guarda_y_carga_sin_cambiar_prediccion(self) -> None:
        modelo = PoliticaMNISTRL.crear(11)
        imagen = np.zeros((28, 28), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "modelo.npz"
            modelo.guardar(ruta)
            restaurado = PoliticaMNISTRL.cargar(ruta)
            prediccion_original = modelo.predecir_imagen(imagen)
            prediccion_restaurada = restaurado.predecir_imagen(imagen)
        self.assertEqual(prediccion_original[0], prediccion_restaurada[0])
        np.testing.assert_allclose(
            prediccion_original[1],
            prediccion_restaurada[1],
        )

    def test_imagen_externa_se_centra_y_convierte_a_28(self) -> None:
        imagen = Image.new("L", (100, 80), "white")
        dibujo = ImageDraw.Draw(imagen)
        dibujo.line((50, 12, 50, 68), fill="black", width=12)
        procesada = normalizar_imagen_externa(imagen)
        self.assertEqual(procesada.shape, (28, 28))
        self.assertEqual(procesada.dtype, np.uint8)
        self.assertGreater(int(procesada.max()), 150)
        self.assertEqual(int(procesada[0].sum()), 0)

    def test_consulta_solo_devuelve_el_numero_solicitado(self) -> None:
        imagenes = np.zeros((30, 28, 28), dtype=np.uint8)
        etiquetas = np.repeat(np.arange(10, dtype=np.uint8), 3)
        indices = seleccionar_digito(imagenes, etiquetas, 7, 3, 99)
        self.assertTrue(np.all(etiquetas[indices] == 7))
        mosaico = construir_mosaico(imagenes, indices)
        self.assertGreater(mosaico.width, 0)
        self.assertGreater(mosaico.height, 0)


if __name__ == "__main__":
    unittest.main()
