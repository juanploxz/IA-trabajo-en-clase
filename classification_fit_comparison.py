"""CLI para estudiar subajuste y sobreajuste en dos datasets clínicos/químicos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from classification_fit.experiment import (
    ejecutar_comparacion_ajuste,
    guardar_complejidad_csv,
    guardar_cv_csv,
    guardar_metricas_csv,
    guardar_predicciones_csv,
    imprimir_resumen,
)

DIRECTORIO_SALIDA = Path("output")


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara subajuste y sobreajuste de K-NN y árboles con Wine y "
            "Breast Cancer usando holdout 50/50, 40/60 y CV estratificada."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Pliegues de validación cruzada; predeterminado 5.",
    )
    parser.add_argument(
        "--reference-k",
        type=int,
        default=7,
        help="Valor k intermedio solicitado; predeterminado 7.",
    )
    parser.add_argument(
        "--high-k",
        type=int,
        default=51,
        help="Valor k grande para observar subajuste; predeterminado 51.",
    )
    parser.add_argument(
        "--train-sizes",
        type=float,
        nargs="+",
        default=(0.50, 0.40),
        help="Proporciones de entrenamiento; predeterminado 0.50 0.40.",
    )
    parser.add_argument("--output-dir", type=Path, default=DIRECTORIO_SALIDA)
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
    comparacion = ejecutar_comparacion_ajuste(
        proporciones_entrenamiento=tuple(argumentos.train_sizes),
        semilla=argumentos.seed,
        pliegues_cv=argumentos.cv_folds,
        k_referencia=argumentos.reference_k,
        k_alto=argumentos.high_k,
    )
    imprimir_resumen(comparacion)
    directorio = argumentos.output_dir
    rutas_csv = (
        directorio / "clasificadores_ajuste_metricas.csv",
        directorio / "clasificadores_ajuste_cv.csv",
        directorio / "clasificadores_ajuste_complejidad.csv",
        directorio / "clasificadores_ajuste_predicciones.csv",
    )
    guardar_metricas_csv(comparacion, rutas_csv[0])
    guardar_cv_csv(comparacion, rutas_csv[1])
    guardar_complejidad_csv(comparacion, rutas_csv[2])
    guardar_predicciones_csv(comparacion, rutas_csv[3])

    sin_interfaz = argumentos.no_gui or _entorno_sin_pantalla()
    _configurar_backend(sin_interfaz)
    from classification_fit.visualization import generar_visualizaciones

    rutas_png = (
        directorio / "clasificadores_ajuste_metricas.png",
        directorio / "clasificadores_ajuste_knn.png",
        directorio / "clasificadores_ajuste_arbol.png",
    )
    generar_visualizaciones(
        comparacion,
        *rutas_png,
        mostrar_ventana=not sin_interfaz,
    )
    print("\nArchivos generados:")
    for ruta in (*rutas_png, *rutas_csv):
        print(f"- {ruta.resolve()}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argumentos = crear_parser().parse_args()
    try:
        return ejecutar(argumentos)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
