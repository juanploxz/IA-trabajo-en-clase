"""CLI independiente para comparar regresores con California Housing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from housing_regression.experiment import (
    ejecutar_experimento_regresion,
    generar_lineas_estimacion,
    guardar_cv_detalle_csv,
    guardar_lineas_estimacion_csv,
    guardar_metricas_csv,
    guardar_predicciones_csv,
    imprimir_resumen,
)

DIRECTORIO_SALIDA = Path("output")
DIRECTORIO_DATOS = Path("data/california_housing")


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara regresión lineal, polinomial, log-lineal y árbol de "
            "decisión sobre California Housing."
        )
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=10)
    parser.add_argument("--tree-min-leaf", type=int, default=5)
    parser.add_argument(
        "--line-feature",
        type=int,
        default=0,
        help="Índice del atributo para las líneas; predeterminado 0 (ingreso).",
    )
    parser.add_argument("--data-dir", type=Path, default=DIRECTORIO_DATOS)
    parser.add_argument("--output-dir", type=Path, default=DIRECTORIO_SALIDA)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Falla si el dataset todavía no existe en el caché local.",
    )
    parser.add_argument("--no-gui", action="store_true")
    return parser


def _entorno_sin_pantalla() -> bool:
    if platform.system() in {"Windows", "Darwin"}:
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _configurar_backend(sin_interfaz: bool) -> None:
    ruta_config = Path(".matplotlib-cache").resolve()
    ruta_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ruta_config))
    if sin_interfaz:
        import matplotlib

        matplotlib.use("Agg", force=True)


def ejecutar(argumentos: argparse.Namespace) -> int:
    profundidad = argumentos.tree_depth
    resultado = ejecutar_experimento_regresion(
        proporcion_prueba=argumentos.test_size,
        semilla=argumentos.seed,
        pliegues_cv=argumentos.cv_folds,
        grado_polinomial=argumentos.poly_degree,
        profundidad_arbol=profundidad,
        minimo_hoja=argumentos.tree_min_leaf,
        directorio_datos=argumentos.data_dir,
        descargar_si_falta=not argumentos.no_download,
    )
    imprimir_resumen(resultado)
    lineas = generar_lineas_estimacion(
        resultado,
        indice_atributo=argumentos.line_feature,
    )

    directorio = argumentos.output_dir
    ruta_metricas_png = directorio / "california_regresion_metricas.png"
    ruta_estimaciones_png = directorio / "california_regresion_estimaciones.png"
    ruta_metricas_csv = directorio / "california_regresion_metricas.csv"
    ruta_cv_csv = directorio / "california_regresion_cv.csv"
    ruta_predicciones_csv = directorio / "california_regresion_predicciones.csv"
    ruta_lineas_csv = directorio / "california_regresion_lineas.csv"
    guardar_metricas_csv(resultado, ruta_metricas_csv)
    guardar_cv_detalle_csv(resultado, ruta_cv_csv)
    guardar_predicciones_csv(resultado, ruta_predicciones_csv)
    guardar_lineas_estimacion_csv(lineas, ruta_lineas_csv)

    sin_interfaz = argumentos.no_gui or _entorno_sin_pantalla()
    _configurar_backend(sin_interfaz)
    from housing_regression.visualization import generar_visualizaciones

    generar_visualizaciones(
        resultado,
        lineas,
        ruta_metricas_png,
        ruta_estimaciones_png,
        mostrar_ventana=not sin_interfaz,
    )
    print("\nArchivos generados:")
    for ruta in (
        ruta_metricas_png,
        ruta_estimaciones_png,
        ruta_metricas_csv,
        ruta_cv_csv,
        ruta_predicciones_csv,
        ruta_lineas_csv,
    ):
        print(f"- {ruta.resolve()}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argumentos = crear_parser().parse_args()
    try:
        return ejecutar(argumentos)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

