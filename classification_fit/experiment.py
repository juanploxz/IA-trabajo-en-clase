"""Experimentos de subajuste y sobreajuste con Wine y Breast Cancer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores",
    category=UserWarning,
)

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROPORCIONES_ENTRENAMIENTO = (0.50, 0.40)
VALORES_K_CURVA = (1, 3, 5, 7, 11, 15, 21, 31, 41, 51)
PROFUNDIDADES_CURVA: tuple[int | None, ...] = (1, 2, 3, 5, 8, None)


@dataclass(slots=True)
class DatasetClasificacion:
    """Datos y metadatos necesarios para un experimento."""

    clave: str
    nombre: str
    datos: np.ndarray
    etiquetas: np.ndarray
    nombres_atributos: tuple[str, ...]
    nombres_clases: tuple[str, ...]


@dataclass(slots=True)
class ResultadoModeloAjuste:
    """Modelo entrenado y evidencias de entrenamiento, CV y prueba."""

    nombre: str
    familia: str
    tendencia_esperada: str
    modelo: Any
    predicciones: np.ndarray
    metricas_entrenamiento: dict[str, float]
    metricas_prueba: dict[str, float]
    cv_entrenamiento_accuracy: np.ndarray
    cv_validacion_accuracy: np.ndarray
    cv_validacion_f1_macro: np.ndarray
    matriz_confusion: np.ndarray
    diagnostico_observado: str = "Pendiente"

    @property
    def brecha_prueba(self) -> float:
        return (
            self.metricas_entrenamiento["accuracy"]
            - self.metricas_prueba["accuracy"]
        )

    @property
    def brecha_cv(self) -> float:
        return float(
            np.mean(self.cv_entrenamiento_accuracy)
            - np.mean(self.cv_validacion_accuracy)
        )


@dataclass(slots=True)
class CurvaComplejidad:
    """Exactitud al recorrer un hiperparámetro de complejidad."""

    familia: str
    parametros: tuple[int | None, ...]
    etiquetas: tuple[str, ...]
    accuracy_entrenamiento: np.ndarray
    accuracy_prueba: np.ndarray
    cv_entrenamiento_media: np.ndarray
    cv_validacion_media: np.ndarray
    cv_validacion_desviacion: np.ndarray


@dataclass(slots=True)
class ResultadoParticionAjuste:
    """Resultados de un dataset en una partición estratificada."""

    dataset: DatasetClasificacion
    proporcion_entrenamiento: float
    semilla: int
    pliegues_cv: int
    indices_entrenamiento: np.ndarray
    indices_prueba: np.ndarray
    x_entrenamiento: np.ndarray
    x_prueba: np.ndarray
    y_entrenamiento: np.ndarray
    y_prueba: np.ndarray
    modelos: dict[str, ResultadoModeloAjuste]
    curvas: dict[str, CurvaComplejidad]

    @property
    def etiqueta_particion(self) -> str:
        entrenamiento = round(self.proporcion_entrenamiento * 100)
        return f"{entrenamiento}/{100 - entrenamiento}"


@dataclass(slots=True)
class ResultadoComparacionAjuste:
    """Agrupa ambos datasets y todas las particiones comparadas."""

    particiones: tuple[ResultadoParticionAjuste, ...]
    semilla: int
    pliegues_cv: int
    k_referencia: int
    k_alto: int


def cargar_datasets() -> dict[str, DatasetClasificacion]:
    """Carga los dos conjuntos integrados en scikit-learn, sin internet."""
    wine = load_wine()
    cancer = load_breast_cancer()
    return {
        "wine": DatasetClasificacion(
            clave="wine",
            nombre="Wine",
            datos=np.asarray(wine.data, dtype=np.float64),
            etiquetas=np.asarray(wine.target, dtype=np.int64),
            nombres_atributos=tuple(str(nombre) for nombre in wine.feature_names),
            nombres_clases=("Cultivar 1", "Cultivar 2", "Cultivar 3"),
        ),
        "breast_cancer": DatasetClasificacion(
            clave="breast_cancer",
            nombre="Breast Cancer Wisconsin",
            datos=np.asarray(cancer.data, dtype=np.float64),
            etiquetas=np.asarray(cancer.target, dtype=np.int64),
            nombres_atributos=tuple(str(nombre) for nombre in cancer.feature_names),
            nombres_clases=("Maligno", "Benigno"),
        ),
    }


def _crear_knn(vecinos: int) -> Pipeline:
    return Pipeline(
        (
            ("escalador", StandardScaler()),
            ("clasificador", KNeighborsClassifier(n_neighbors=vecinos)),
        )
    )


def crear_clasificadores(
    k_referencia: int = 7,
    k_alto: int = 51,
    semilla: int = 42,
) -> dict[str, tuple[Any, str, str]]:
    """Crea configuraciones deliberadamente simples, intermedias y flexibles."""
    if k_referencia <= 1:
        raise ValueError("El k de referencia debe ser mayor que 1.")
    if k_alto <= k_referencia:
        raise ValueError("El k alto debe ser mayor que el k de referencia.")
    return {
        "K-NN (k=1)": (
            _crear_knn(1),
            "K-NN",
            "Sobreajuste probable",
        ),
        f"K-NN (k={k_referencia})": (
            _crear_knn(k_referencia),
            "K-NN",
            "Referencia solicitada",
        ),
        f"K-NN (k={k_alto})": (
            _crear_knn(k_alto),
            "K-NN",
            "Subajuste probable",
        ),
        "Árbol (profundidad 1)": (
            DecisionTreeClassifier(max_depth=1, random_state=semilla),
            "Árbol de decisión",
            "Subajuste probable",
        ),
        "Árbol (profundidad 5)": (
            DecisionTreeClassifier(max_depth=5, random_state=semilla),
            "Árbol de decisión",
            "Complejidad intermedia",
        ),
        "Árbol (sin límite)": (
            DecisionTreeClassifier(max_depth=None, random_state=semilla),
            "Árbol de decisión",
            "Sobreajuste probable",
        ),
    }


def _metricas(etiquetas: np.ndarray, predicciones: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(etiquetas, predicciones)),
        "balanced_accuracy": float(
            balanced_accuracy_score(etiquetas, predicciones)
        ),
        "precision_macro": float(
            precision_score(
                etiquetas,
                predicciones,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                etiquetas,
                predicciones,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                etiquetas,
                predicciones,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _validar_k_para_cv(
    k_alto: int,
    y_entrenamiento: np.ndarray,
    pliegues_cv: int,
) -> None:
    cantidad = len(y_entrenamiento)
    mayor_pliegue_validacion = int(np.ceil(cantidad / pliegues_cv))
    minimo_entrenamiento_cv = cantidad - mayor_pliegue_validacion
    if k_alto > minimo_entrenamiento_cv:
        raise ValueError(
            f"k={k_alto} excede las {minimo_entrenamiento_cv} muestras "
            "disponibles en el menor entrenamiento interno de CV."
        )


def _evaluar_modelo(
    nombre: str,
    modelo: Any,
    familia: str,
    tendencia: str,
    x_entrenamiento: np.ndarray,
    x_prueba: np.ndarray,
    y_entrenamiento: np.ndarray,
    y_prueba: np.ndarray,
    validacion: StratifiedKFold,
    cantidad_clases: int,
) -> ResultadoModeloAjuste:
    puntuaciones = cross_validate(
        clone(modelo),
        x_entrenamiento,
        y_entrenamiento,
        cv=validacion,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        return_train_score=True,
        error_score="raise",
    )
    modelo_ajustado = clone(modelo)
    modelo_ajustado.fit(x_entrenamiento, y_entrenamiento)
    predicciones_entrenamiento = np.asarray(
        modelo_ajustado.predict(x_entrenamiento),
        dtype=np.int64,
    )
    predicciones_prueba = np.asarray(
        modelo_ajustado.predict(x_prueba),
        dtype=np.int64,
    )
    return ResultadoModeloAjuste(
        nombre=nombre,
        familia=familia,
        tendencia_esperada=tendencia,
        modelo=modelo_ajustado,
        predicciones=predicciones_prueba,
        metricas_entrenamiento=_metricas(
            y_entrenamiento,
            predicciones_entrenamiento,
        ),
        metricas_prueba=_metricas(y_prueba, predicciones_prueba),
        cv_entrenamiento_accuracy=np.asarray(
            puntuaciones["train_accuracy"],
            dtype=np.float64,
        ),
        cv_validacion_accuracy=np.asarray(
            puntuaciones["test_accuracy"],
            dtype=np.float64,
        ),
        cv_validacion_f1_macro=np.asarray(
            puntuaciones["test_f1_macro"],
            dtype=np.float64,
        ),
        matriz_confusion=confusion_matrix(
            y_prueba,
            predicciones_prueba,
            labels=np.arange(cantidad_clases),
        ),
    )


def _diagnosticar_modelos(modelos: dict[str, ResultadoModeloAjuste]) -> None:
    """Clasifica patrones relativos; no reemplaza una prueba estadística."""
    mejor_cv = max(
        float(np.mean(modelo.cv_validacion_accuracy))
        for modelo in modelos.values()
    )
    for modelo in modelos.values():
        entrenamiento_cv = float(np.mean(modelo.cv_entrenamiento_accuracy))
        validacion_cv = float(np.mean(modelo.cv_validacion_accuracy))
        if (
            validacion_cv <= mejor_cv - 0.03
            and (
                modelo.tendencia_esperada == "Subajuste probable"
                or entrenamiento_cv <= mejor_cv - 0.02
            )
        ):
            modelo.diagnostico_observado = "Subajuste observado"
        elif (
            entrenamiento_cv - validacion_cv >= 0.05
            or modelo.brecha_prueba >= 0.08
        ):
            modelo.diagnostico_observado = "Sobreajuste observado"
        else:
            modelo.diagnostico_observado = "Ajuste competitivo"


def _crear_curva(
    familia: str,
    parametros: tuple[int | None, ...],
    etiquetas: tuple[str, ...],
    creador: Any,
    x_entrenamiento: np.ndarray,
    x_prueba: np.ndarray,
    y_entrenamiento: np.ndarray,
    y_prueba: np.ndarray,
    validacion: StratifiedKFold,
) -> CurvaComplejidad:
    accuracy_entrenamiento: list[float] = []
    accuracy_prueba: list[float] = []
    cv_entrenamiento_media: list[float] = []
    cv_validacion_media: list[float] = []
    cv_validacion_desviacion: list[float] = []
    for parametro in parametros:
        modelo = creador(parametro)
        puntuaciones = cross_validate(
            modelo,
            x_entrenamiento,
            y_entrenamiento,
            cv=validacion,
            scoring="accuracy",
            return_train_score=True,
            error_score="raise",
        )
        modelo.fit(x_entrenamiento, y_entrenamiento)
        accuracy_entrenamiento.append(
            float(modelo.score(x_entrenamiento, y_entrenamiento))
        )
        accuracy_prueba.append(float(modelo.score(x_prueba, y_prueba)))
        cv_entrenamiento_media.append(float(np.mean(puntuaciones["train_score"])))
        cv_validacion_media.append(float(np.mean(puntuaciones["test_score"])))
        cv_validacion_desviacion.append(
            float(np.std(puntuaciones["test_score"], ddof=1))
        )
    return CurvaComplejidad(
        familia=familia,
        parametros=parametros,
        etiquetas=etiquetas,
        accuracy_entrenamiento=np.asarray(accuracy_entrenamiento),
        accuracy_prueba=np.asarray(accuracy_prueba),
        cv_entrenamiento_media=np.asarray(cv_entrenamiento_media),
        cv_validacion_media=np.asarray(cv_validacion_media),
        cv_validacion_desviacion=np.asarray(cv_validacion_desviacion),
    )


def _crear_curvas(
    x_entrenamiento: np.ndarray,
    x_prueba: np.ndarray,
    y_entrenamiento: np.ndarray,
    y_prueba: np.ndarray,
    validacion: StratifiedKFold,
    semilla: int,
    k_referencia: int,
    k_alto: int,
) -> dict[str, CurvaComplejidad]:
    valores_k = tuple(
        sorted(
            {
                valor
                for valor in (*VALORES_K_CURVA, k_referencia, k_alto)
                if valor <= k_alto
            }
        )
    )
    curva_knn = _crear_curva(
        "K-NN",
        valores_k,
        tuple(f"k={valor}" for valor in valores_k),
        _crear_knn,
        x_entrenamiento,
        x_prueba,
        y_entrenamiento,
        y_prueba,
        validacion,
    )
    curva_arbol = _crear_curva(
        "Árbol de decisión",
        PROFUNDIDADES_CURVA,
        tuple(
            "Sin límite" if profundidad is None else str(profundidad)
            for profundidad in PROFUNDIDADES_CURVA
        ),
        lambda profundidad: DecisionTreeClassifier(
            max_depth=profundidad,
            random_state=semilla,
        ),
        x_entrenamiento,
        x_prueba,
        y_entrenamiento,
        y_prueba,
        validacion,
    )
    return {"K-NN": curva_knn, "Árbol de decisión": curva_arbol}


def ejecutar_comparacion_ajuste(
    proporciones_entrenamiento: tuple[float, ...] = PROPORCIONES_ENTRENAMIENTO,
    claves_datasets: tuple[str, ...] = ("wine", "breast_cancer"),
    semilla: int = 42,
    pliegues_cv: int = 5,
    k_referencia: int = 7,
    k_alto: int = 51,
    incluir_curvas: bool = True,
) -> ResultadoComparacionAjuste:
    """Ejecuta holdout estratificado y CV solo sobre el entrenamiento."""
    if not proporciones_entrenamiento:
        raise ValueError("Debe indicarse al menos una partición.")
    if len(set(proporciones_entrenamiento)) != len(proporciones_entrenamiento):
        raise ValueError("Las particiones no deben repetirse.")
    if pliegues_cv < 2:
        raise ValueError("La validación cruzada requiere al menos dos pliegues.")
    datasets = cargar_datasets()
    claves_invalidas = set(claves_datasets) - set(datasets)
    if claves_invalidas:
        raise ValueError(f"Datasets desconocidos: {sorted(claves_invalidas)}")

    resultados: list[ResultadoParticionAjuste] = []
    for clave in claves_datasets:
        dataset = datasets[clave]
        indices = np.arange(len(dataset.datos), dtype=np.int64)
        for proporcion_entrenamiento in proporciones_entrenamiento:
            if not 0.0 < proporcion_entrenamiento < 1.0:
                raise ValueError("Cada proporción debe pertenecer a (0, 1).")
            (
                x_entrenamiento,
                x_prueba,
                y_entrenamiento,
                y_prueba,
                indices_entrenamiento,
                indices_prueba,
            ) = train_test_split(
                dataset.datos,
                dataset.etiquetas,
                indices,
                train_size=proporcion_entrenamiento,
                random_state=semilla,
                stratify=dataset.etiquetas,
            )
            if np.min(np.bincount(y_entrenamiento)) < pliegues_cv:
                raise ValueError(
                    "Cada clase de entrenamiento debe tener al menos una "
                    "muestra por pliegue."
                )
            _validar_k_para_cv(k_alto, y_entrenamiento, pliegues_cv)
            validacion = StratifiedKFold(
                n_splits=pliegues_cv,
                shuffle=True,
                random_state=semilla,
            )
            modelos: dict[str, ResultadoModeloAjuste] = {}
            for nombre, (modelo, familia, tendencia) in crear_clasificadores(
                k_referencia,
                k_alto,
                semilla,
            ).items():
                modelos[nombre] = _evaluar_modelo(
                    nombre,
                    modelo,
                    familia,
                    tendencia,
                    x_entrenamiento,
                    x_prueba,
                    y_entrenamiento,
                    y_prueba,
                    validacion,
                    len(dataset.nombres_clases),
                )
            _diagnosticar_modelos(modelos)
            curvas = (
                _crear_curvas(
                    x_entrenamiento,
                    x_prueba,
                    y_entrenamiento,
                    y_prueba,
                    validacion,
                    semilla,
                    k_referencia,
                    k_alto,
                )
                if incluir_curvas
                else {}
            )
            resultados.append(
                ResultadoParticionAjuste(
                    dataset=dataset,
                    proporcion_entrenamiento=proporcion_entrenamiento,
                    semilla=semilla,
                    pliegues_cv=pliegues_cv,
                    indices_entrenamiento=indices_entrenamiento,
                    indices_prueba=indices_prueba,
                    x_entrenamiento=x_entrenamiento,
                    x_prueba=x_prueba,
                    y_entrenamiento=y_entrenamiento,
                    y_prueba=y_prueba,
                    modelos=modelos,
                    curvas=curvas,
                )
            )
    return ResultadoComparacionAjuste(
        particiones=tuple(resultados),
        semilla=semilla,
        pliegues_cv=pliegues_cv,
        k_referencia=k_referencia,
        k_alto=k_alto,
    )


def guardar_metricas_csv(
    comparacion: ResultadoComparacionAjuste,
    ruta: Path,
) -> None:
    """Exporta una fila por dataset, partición y modelo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "dataset",
                "particion_entrenamiento_prueba",
                "muestras_entrenamiento",
                "muestras_prueba",
                "modelo",
                "familia",
                "tendencia_esperada",
                "diagnostico_observado",
                "accuracy_entrenamiento",
                "accuracy_prueba",
                "brecha_prueba",
                "balanced_accuracy_prueba",
                "precision_macro_prueba",
                "recall_macro_prueba",
                "f1_macro_prueba",
                "accuracy_cv_entrenamiento_media",
                "accuracy_cv_validacion_media",
                "accuracy_cv_validacion_desviacion",
                "brecha_cv",
                "f1_macro_cv_validacion_media",
                "f1_macro_cv_validacion_desviacion",
            )
        )
        for particion in comparacion.particiones:
            for modelo in particion.modelos.values():
                escritor.writerow(
                    (
                        particion.dataset.nombre,
                        particion.etiqueta_particion,
                        len(particion.x_entrenamiento),
                        len(particion.x_prueba),
                        modelo.nombre,
                        modelo.familia,
                        modelo.tendencia_esperada,
                        modelo.diagnostico_observado,
                        modelo.metricas_entrenamiento["accuracy"],
                        modelo.metricas_prueba["accuracy"],
                        modelo.brecha_prueba,
                        modelo.metricas_prueba["balanced_accuracy"],
                        modelo.metricas_prueba["precision_macro"],
                        modelo.metricas_prueba["recall_macro"],
                        modelo.metricas_prueba["f1_macro"],
                        np.mean(modelo.cv_entrenamiento_accuracy),
                        np.mean(modelo.cv_validacion_accuracy),
                        np.std(modelo.cv_validacion_accuracy, ddof=1),
                        modelo.brecha_cv,
                        np.mean(modelo.cv_validacion_f1_macro),
                        np.std(modelo.cv_validacion_f1_macro, ddof=1),
                    )
                )


def guardar_cv_csv(comparacion: ResultadoComparacionAjuste, ruta: Path) -> None:
    """Exporta los valores individuales de los pliegues de CV."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "dataset",
                "particion_entrenamiento_prueba",
                "modelo",
                "pliegue",
                "accuracy_entrenamiento",
                "accuracy_validacion",
                "f1_macro_validacion",
            )
        )
        for particion in comparacion.particiones:
            for modelo in particion.modelos.values():
                for indice in range(particion.pliegues_cv):
                    escritor.writerow(
                        (
                            particion.dataset.nombre,
                            particion.etiqueta_particion,
                            modelo.nombre,
                            indice + 1,
                            modelo.cv_entrenamiento_accuracy[indice],
                            modelo.cv_validacion_accuracy[indice],
                            modelo.cv_validacion_f1_macro[indice],
                        )
                    )


def guardar_complejidad_csv(
    comparacion: ResultadoComparacionAjuste,
    ruta: Path,
) -> None:
    """Exporta todos los puntos de las curvas de K-NN y árbol."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "dataset",
                "particion_entrenamiento_prueba",
                "familia",
                "parametro",
                "etiqueta_parametro",
                "accuracy_entrenamiento",
                "accuracy_prueba",
                "accuracy_cv_entrenamiento_media",
                "accuracy_cv_validacion_media",
                "accuracy_cv_validacion_desviacion",
            )
        )
        for particion in comparacion.particiones:
            for curva in particion.curvas.values():
                for indice, parametro in enumerate(curva.parametros):
                    escritor.writerow(
                        (
                            particion.dataset.nombre,
                            particion.etiqueta_particion,
                            curva.familia,
                            "none" if parametro is None else parametro,
                            curva.etiquetas[indice],
                            curva.accuracy_entrenamiento[indice],
                            curva.accuracy_prueba[indice],
                            curva.cv_entrenamiento_media[indice],
                            curva.cv_validacion_media[indice],
                            curva.cv_validacion_desviacion[indice],
                        )
                    )


def guardar_predicciones_csv(
    comparacion: ResultadoComparacionAjuste,
    ruta: Path,
) -> None:
    """Exporta clases reales y predicciones de cada conjunto de prueba."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_modelos = tuple(comparacion.particiones[0].modelos)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "dataset",
                "particion_entrenamiento_prueba",
                "indice_original",
                "clase_real",
                *nombres_modelos,
            )
        )
        for particion in comparacion.particiones:
            for posicion, indice_original in enumerate(particion.indices_prueba):
                escritor.writerow(
                    (
                        particion.dataset.nombre,
                        particion.etiqueta_particion,
                        int(indice_original),
                        particion.dataset.nombres_clases[
                            int(particion.y_prueba[posicion])
                        ],
                        *(
                            particion.dataset.nombres_clases[
                                int(particion.modelos[nombre].predicciones[posicion])
                            ]
                            for nombre in nombres_modelos
                        ),
                    )
                )


def imprimir_resumen(comparacion: ResultadoComparacionAjuste) -> None:
    """Imprime el contraste entre entrenamiento, prueba y CV."""
    print("=== Subajuste y sobreajuste en clasificación ===")
    print(
        f"CV estratificada: {comparacion.pliegues_cv} pliegues sobre "
        "cada conjunto de entrenamiento"
    )
    print(f"Semilla: {comparacion.semilla}\n")
    for particion in comparacion.particiones:
        print(
            f"--- {particion.dataset.nombre} | "
            f"partición {particion.etiqueta_particion} ---"
        )
        print(
            f"Entrenamiento={len(particion.x_entrenamiento)}, "
            f"prueba={len(particion.x_prueba)}"
        )
        encabezado = (
            f"{'Modelo':<25} {'Train':>7} {'Test':>7} {'CV':>7} "
            f"{'Brecha CV':>10}  Diagnóstico"
        )
        print(encabezado)
        print("-" * len(encabezado))
        for modelo in particion.modelos.values():
            print(
                f"{modelo.nombre:<25} "
                f"{modelo.metricas_entrenamiento['accuracy']:>7.3f} "
                f"{modelo.metricas_prueba['accuracy']:>7.3f} "
                f"{np.mean(modelo.cv_validacion_accuracy):>7.3f} "
                f"{modelo.brecha_cv:>10.3f}  "
                f"{modelo.diagnostico_observado}"
            )
        print()
    print(
        "Nota: 50/50 y 40/60 son particiones holdout. La validación "
        "cruzada se aplica después y solo al entrenamiento."
    )
