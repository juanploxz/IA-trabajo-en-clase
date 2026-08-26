"""CLI independiente para comparar particiones 60/40, 70/30 y 80/20."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from iris_classifier.split_comparison import (
    ejecutar_comparacion_particiones,
    guardar_metricas_particiones_csv,
    guardar_predicciones_particiones_csv,
    imprimir_resumen_particiones,
)

DIRECTORIO_SALIDA = Path("output")


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara LDA, K-NN(3) y árbol de decisión en Iris con "
            "particiones 60/40, 70/30 y 80/20."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help="Cantidad k de vecinos; predeterminado 3.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=None,
        help="Profundidad máxima; sin límite de forma predeterminada.",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs=2,
        default=(2, 3),
        metavar=("X", "Y"),
        help="Atributos de las fronteras; predeterminado: 2 3.",
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
    comparacion = ejecutar_comparacion_particiones(
        semilla=argumentos.seed,
        vecinos=argumentos.neighbors,
        profundidad_arbol=argumentos.tree_max_depth,
    )
    imprimir_resumen_particiones(comparacion)

    directorio = argumentos.output_dir
    ruta_metricas_png = directorio / "iris_particiones_metricas.png"
    ruta_fronteras_png = directorio / "iris_particiones_fronteras.png"
    ruta_metricas_csv = directorio / "iris_particiones_metricas.csv"
    ruta_predicciones_csv = directorio / "iris_particiones_predicciones.csv"
    guardar_metricas_particiones_csv(comparacion, ruta_metricas_csv)
    guardar_predicciones_particiones_csv(comparacion, ruta_predicciones_csv)

    sin_interfaz = argumentos.no_gui or _entorno_sin_pantalla()
    _configurar_backend(sin_interfaz)
    from iris_classifier.split_visualization import (
        generar_visualizaciones_particiones,
    )

    generar_visualizaciones_particiones(
        comparacion,
        tuple(argumentos.features),
        ruta_metricas_png,
        ruta_fronteras_png,
        mostrar_ventana=not sin_interfaz,
    )
    print("\nArchivos generados:")
    for ruta in (
        ruta_metricas_png,
        ruta_fronteras_png,
        ruta_metricas_csv,
        ruta_predicciones_csv,
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
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

