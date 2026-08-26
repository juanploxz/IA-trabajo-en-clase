"""API pública del ejercicio independiente de clasificación Iris."""

from iris_classifier.experiment import (
    NOMBRES_ATRIBUTOS_ES,
    NOMBRES_CLASES,
    ResultadoClasificador,
    ResultadoExperimento,
    crear_clasificadores,
    ejecutar_experimento,
    entrenar_modelos_frontera,
)

__all__ = [
    "NOMBRES_ATRIBUTOS_ES",
    "NOMBRES_CLASES",
    "ResultadoClasificador",
    "ResultadoExperimento",
    "crear_clasificadores",
    "ejecutar_experimento",
    "entrenar_modelos_frontera",
]
