"""Datos, modelos, validación cruzada y métricas de California Housing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

NOMBRES_ATRIBUTOS_ES = (
    "Ingreso mediano (decenas de miles de USD)",
    "Edad mediana de las viviendas (años)",
    "Habitaciones promedio por hogar",
    "Dormitorios promedio por hogar",
    "Población del bloque censal",
    "Ocupantes promedio por hogar",
    "Latitud",
    "Longitud",
)
UNIDAD_OBJETIVO = "Valor mediano de vivienda (cientos de miles de USD)"


@dataclass(slots=True)
class ResultadoRegresor:
    """Modelo ajustado, predicciones y resultados de evaluación."""

    nombre: str
    modelo: Any
    predicciones: np.ndarray
    metricas_prueba: dict[str, float]
    metricas_cv: dict[str, np.ndarray]
    tiempo_ajuste_segundos: float
    tiempo_cv_medio_segundos: float


@dataclass(slots=True)
class ResultadoExperimentoRegresion:
    """Datos y resultados de los cuatro regresores comparados."""

    datos: np.ndarray
    objetivo: np.ndarray
    indices_entrenamiento: np.ndarray
    indices_prueba: np.ndarray
    x_entrenamiento: np.ndarray
    x_prueba: np.ndarray
    y_entrenamiento: np.ndarray
    y_prueba: np.ndarray
    nombres_atributos: tuple[str, ...]
    regresores: dict[str, ResultadoRegresor]
    proporcion_prueba: float
    semilla: int
    pliegues_cv: int


@dataclass(slots=True)
class LineasEstimacion:
    """Predicciones al variar un atributo y fijar los demás en su mediana."""

    indice_atributo: int
    nombre_atributo: str
    valores_atributo: np.ndarray
    predicciones: dict[str, np.ndarray]
    valores_base: np.ndarray


def cargar_california_housing(
    directorio_datos: Path = Path("data/california_housing"),
    descargar_si_falta: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Descarga o carga California Housing usando el caché del proyecto."""
    directorio_datos.mkdir(parents=True, exist_ok=True)
    conjunto = fetch_california_housing(
        data_home=str(directorio_datos),
        download_if_missing=descargar_si_falta,
    )
    datos = np.asarray(conjunto.data, dtype=np.float64)
    objetivo = np.asarray(conjunto.target, dtype=np.float64)
    return datos, objetivo, NOMBRES_ATRIBUTOS_ES


def crear_regresores(
    grado_polinomial: int = 2,
    profundidad_arbol: int | None = 10,
    minimo_hoja: int = 5,
    semilla: int = 42,
) -> dict[str, Any]:
    """Construye los cuatro modelos con una configuración reproducible."""
    if grado_polinomial < 2:
        raise ValueError("El grado polinomial debe ser al menos 2.")
    if profundidad_arbol is not None and profundidad_arbol <= 0:
        raise ValueError("La profundidad del árbol debe ser positiva.")
    if minimo_hoja <= 0:
        raise ValueError("El mínimo de muestras por hoja debe ser positivo.")

    base_lineal = Pipeline(
        (
            ("escalador", StandardScaler()),
            ("regresor", LinearRegression()),
        )
    )
    modelo_log_lineal = TransformedTargetRegressor(
        regressor=Pipeline(
            (
                ("escalador", StandardScaler()),
                ("regresor", LinearRegression()),
            )
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return {
        "Regresión lineal": base_lineal,
        f"Regresión polinomial (grado {grado_polinomial})": Pipeline(
            (
                ("escalador", StandardScaler()),
                (
                    "caracteristicas_polinomiales",
                    PolynomialFeatures(
                        degree=grado_polinomial,
                        include_bias=False,
                    ),
                ),
                ("regresor", LinearRegression()),
            )
        ),
        "Regresión log-lineal": modelo_log_lineal,
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=profundidad_arbol,
            min_samples_leaf=minimo_hoja,
            random_state=semilla,
        ),
    }


def _validar_datos(
    datos: np.ndarray,
    objetivo: np.ndarray,
    nombres_atributos: tuple[str, ...],
) -> None:
    if datos.ndim != 2:
        raise ValueError("Los datos deben formar una matriz bidimensional.")
    if objetivo.ndim != 1:
        raise ValueError("El objetivo debe ser un vector unidimensional.")
    if len(datos) != len(objetivo) or len(datos) < 20:
        raise ValueError("Datos y objetivo deben tener al menos 20 filas coincidentes.")
    if datos.shape[1] != len(nombres_atributos):
        raise ValueError("Debe existir un nombre por cada atributo.")
    if not np.all(np.isfinite(datos)) or not np.all(np.isfinite(objetivo)):
        raise ValueError("Los datos no deben contener NaN ni infinitos.")
    if np.any(objetivo < 0.0):
        raise ValueError("La regresión log-lineal requiere un objetivo no negativo.")


def ejecutar_experimento_regresion(
    datos: np.ndarray | None = None,
    objetivo: np.ndarray | None = None,
    nombres_atributos: tuple[str, ...] | None = None,
    proporcion_prueba: float = 0.20,
    semilla: int = 42,
    pliegues_cv: int = 5,
    grado_polinomial: int = 2,
    profundidad_arbol: int | None = 10,
    minimo_hoja: int = 5,
    directorio_datos: Path = Path("data/california_housing"),
    descargar_si_falta: bool = True,
) -> ResultadoExperimentoRegresion:
    """Compara holdout 80/20 y CV sobre entrenamiento para cuatro modelos."""
    if (datos is None) != (objetivo is None):
        raise ValueError("Deben proporcionarse juntos los datos y el objetivo.")
    if datos is None or objetivo is None:
        datos, objetivo, nombres_cargados = cargar_california_housing(
            directorio_datos,
            descargar_si_falta,
        )
        nombres_atributos = nombres_cargados
    else:
        datos = np.asarray(datos, dtype=np.float64)
        objetivo = np.asarray(objetivo, dtype=np.float64)
        if nombres_atributos is None:
            nombres_atributos = tuple(
                f"Atributo {indice + 1}" for indice in range(datos.shape[1])
            )
    nombres_atributos = tuple(nombres_atributos)
    _validar_datos(datos, objetivo, nombres_atributos)
    if not 0.0 < proporcion_prueba < 1.0:
        raise ValueError("La proporción de prueba debe pertenecer a (0, 1).")
    if pliegues_cv < 2:
        raise ValueError("La validación cruzada requiere al menos dos pliegues.")

    indices = np.arange(len(datos), dtype=np.int64)
    (
        x_entrenamiento,
        x_prueba,
        y_entrenamiento,
        y_prueba,
        indices_entrenamiento,
        indices_prueba,
    ) = train_test_split(
        datos,
        objetivo,
        indices,
        test_size=proporcion_prueba,
        random_state=semilla,
    )
    if pliegues_cv > len(x_entrenamiento):
        raise ValueError("Hay más pliegues que muestras de entrenamiento.")

    validacion = KFold(
        n_splits=pliegues_cv,
        shuffle=True,
        random_state=semilla,
    )
    puntuaciones = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    resultados: dict[str, ResultadoRegresor] = {}
    for nombre, modelo in crear_regresores(
        grado_polinomial,
        profundidad_arbol,
        minimo_hoja,
        semilla,
    ).items():
        resultados_cv = cross_validate(
            modelo,
            x_entrenamiento,
            y_entrenamiento,
            cv=validacion,
            scoring=puntuaciones,
            return_train_score=False,
        )
        inicio = perf_counter()
        modelo.fit(x_entrenamiento, y_entrenamiento)
        tiempo_ajuste = perf_counter() - inicio
        predicciones = np.asarray(modelo.predict(x_prueba), dtype=np.float64)
        resultados[nombre] = ResultadoRegresor(
            nombre=nombre,
            modelo=modelo,
            predicciones=predicciones,
            metricas_prueba={
                "mae": float(mean_absolute_error(y_prueba, predicciones)),
                "rmse": float(root_mean_squared_error(y_prueba, predicciones)),
                "r2": float(r2_score(y_prueba, predicciones)),
            },
            metricas_cv={
                "mae": -np.asarray(resultados_cv["test_mae"], dtype=np.float64),
                "rmse": -np.asarray(
                    resultados_cv["test_rmse"],
                    dtype=np.float64,
                ),
                "r2": np.asarray(resultados_cv["test_r2"], dtype=np.float64),
            },
            tiempo_ajuste_segundos=tiempo_ajuste,
            tiempo_cv_medio_segundos=float(np.mean(resultados_cv["fit_time"])),
        )

    return ResultadoExperimentoRegresion(
        datos=datos,
        objetivo=objetivo,
        indices_entrenamiento=indices_entrenamiento,
        indices_prueba=indices_prueba,
        x_entrenamiento=x_entrenamiento,
        x_prueba=x_prueba,
        y_entrenamiento=y_entrenamiento,
        y_prueba=y_prueba,
        nombres_atributos=nombres_atributos,
        regresores=resultados,
        proporcion_prueba=proporcion_prueba,
        semilla=semilla,
        pliegues_cv=pliegues_cv,
    )


def generar_lineas_estimacion(
    resultado: ResultadoExperimentoRegresion,
    indice_atributo: int = 0,
    cantidad_puntos: int = 250,
) -> LineasEstimacion:
    """Predice sobre una rejilla unidimensional con los demás atributos fijos."""
    if indice_atributo not in range(resultado.datos.shape[1]):
        raise ValueError("El índice del atributo está fuera de rango.")
    if cantidad_puntos < 20:
        raise ValueError("Se requieren al menos 20 puntos para la línea.")
    minimo, maximo = np.quantile(
        resultado.x_entrenamiento[:, indice_atributo],
        (0.01, 0.99),
    )
    valores = np.linspace(float(minimo), float(maximo), cantidad_puntos)
    base = np.median(resultado.x_entrenamiento, axis=0)
    matriz = np.tile(base, (cantidad_puntos, 1))
    matriz[:, indice_atributo] = valores
    predicciones = {
        nombre: np.asarray(regresor.modelo.predict(matriz), dtype=np.float64)
        for nombre, regresor in resultado.regresores.items()
    }
    return LineasEstimacion(
        indice_atributo=indice_atributo,
        nombre_atributo=resultado.nombres_atributos[indice_atributo],
        valores_atributo=valores,
        predicciones=predicciones,
        valores_base=base,
    )


def resumir_validacion(
    regresor: ResultadoRegresor,
    metrica: str,
) -> tuple[float, float]:
    """Devuelve media y desviación estándar muestral de una métrica CV."""
    valores = regresor.metricas_cv[metrica]
    desviacion = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
    return float(np.mean(valores)), desviacion


def guardar_metricas_csv(
    resultado: ResultadoExperimentoRegresion,
    ruta: Path,
) -> None:
    """Exporta una fila resumen por modelo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "modelo",
                "mae_prueba",
                "rmse_prueba",
                "r2_prueba",
                "mae_cv_media",
                "mae_cv_desviacion",
                "rmse_cv_media",
                "rmse_cv_desviacion",
                "r2_cv_media",
                "r2_cv_desviacion",
                "tiempo_ajuste_segundos",
                "tiempo_cv_medio_segundos",
            )
        )
        for regresor in resultado.regresores.values():
            mae_cv = resumir_validacion(regresor, "mae")
            rmse_cv = resumir_validacion(regresor, "rmse")
            r2_cv = resumir_validacion(regresor, "r2")
            escritor.writerow(
                (
                    regresor.nombre,
                    regresor.metricas_prueba["mae"],
                    regresor.metricas_prueba["rmse"],
                    regresor.metricas_prueba["r2"],
                    *mae_cv,
                    *rmse_cv,
                    *r2_cv,
                    regresor.tiempo_ajuste_segundos,
                    regresor.tiempo_cv_medio_segundos,
                )
            )


def guardar_cv_detalle_csv(
    resultado: ResultadoExperimentoRegresion,
    ruta: Path,
) -> None:
    """Exporta una fila por modelo y pliegue de validación cruzada."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("modelo", "pliegue", "mae", "rmse", "r2"))
        for regresor in resultado.regresores.values():
            for indice in range(resultado.pliegues_cv):
                escritor.writerow(
                    (
                        regresor.nombre,
                        indice + 1,
                        regresor.metricas_cv["mae"][indice],
                        regresor.metricas_cv["rmse"][indice],
                        regresor.metricas_cv["r2"][indice],
                    )
                )


def guardar_predicciones_csv(
    resultado: ResultadoExperimentoRegresion,
    ruta: Path,
) -> None:
    """Exporta atributos, valor real y predicciones del conjunto de prueba."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_modelos = tuple(resultado.regresores)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            (
                "indice_california_housing",
                *resultado.nombres_atributos,
                "valor_real",
                *nombres_modelos,
            )
        )
        for posicion, indice_original in enumerate(resultado.indices_prueba):
            escritor.writerow(
                (
                    int(indice_original),
                    *resultado.x_prueba[posicion].tolist(),
                    resultado.y_prueba[posicion],
                    *(
                        resultado.regresores[nombre].predicciones[posicion]
                        for nombre in nombres_modelos
                    ),
                )
            )


def guardar_lineas_estimacion_csv(
    lineas: LineasEstimacion,
    ruta: Path,
) -> None:
    """Exporta los puntos que forman las líneas de estimación."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_modelos = tuple(lineas.predicciones)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow((lineas.nombre_atributo, *nombres_modelos))
        for indice, valor in enumerate(lineas.valores_atributo):
            escritor.writerow(
                (
                    valor,
                    *(
                        lineas.predicciones[nombre][indice]
                        for nombre in nombres_modelos
                    ),
                )
            )


def imprimir_resumen(resultado: ResultadoExperimentoRegresion) -> None:
    """Presenta tamaños, holdout y validación cruzada en terminal."""
    print("=== Regresión con California Housing ===")
    print(f"Muestras totales: {len(resultado.datos)}")
    print(f"Atributos: {resultado.datos.shape[1]}")
    print(f"Entrenamiento: {len(resultado.x_entrenamiento)} (80%)")
    print(f"Prueba: {len(resultado.x_prueba)} (20%)")
    print(f"Validación cruzada: {resultado.pliegues_cv} pliegues sobre entrenamiento")
    print(f"Semilla: {resultado.semilla}\n")
    encabezado = (
        f"{'Modelo':<34} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
        f"{'R² CV (media ± desv.)':>24}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for regresor in resultado.regresores.values():
        r2_media, r2_desviacion = resumir_validacion(regresor, "r2")
        print(
            f"{regresor.nombre:<34} "
            f"{regresor.metricas_prueba['mae']:>8.4f} "
            f"{regresor.metricas_prueba['rmse']:>8.4f} "
            f"{regresor.metricas_prueba['r2']:>8.4f} "
            f"{r2_media:>10.4f} ± {r2_desviacion:<8.4f}"
        )
    mejor = max(
        resultado.regresores.values(),
        key=lambda item: item.metricas_prueba["r2"],
    )
    print(f"\nMejor R² de prueba: {mejor.nombre} ({mejor.metricas_prueba['r2']:.4f})")
