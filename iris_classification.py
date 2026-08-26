"""CLI independiente para clasificación del conjunto Iris."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from iris_classifier.experiment import (
    ejecutar_experimento,
    guardar_metricas_csv,
    guardar_predicciones_csv,
    imprimir_resumen,
)

DIRECTORIO_SALIDA = Path("output")


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica Iris con LDA, K-NN y árbol de decisión; genera "
            "fronteras y métricas."
        )
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Proporción de prueba; predeterminado 0.30 (partición 70/30).",
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
        help="Índices de los dos atributos para fronteras; predeterminado 2 3.",
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
    atributos = tuple(argumentos.features)
    resultado = ejecutar_experimento(
        proporcion_prueba=argumentos.test_size,
        semilla=argumentos.seed,
        vecinos=argumentos.neighbors,
        profundidad_arbol=argumentos.tree_max_depth,
    )
    imprimir_resumen(resultado)
    directorio = argumentos.output_dir
    ruta_fronteras = directorio / "iris_fronteras_decision.png"
    ruta_metricas = directorio / "iris_metricas.png"
    ruta_csv = directorio / "iris_metricas.csv"
    ruta_predicciones = directorio / "iris_predicciones.csv"
    guardar_metricas_csv(resultado, ruta_csv)
    guardar_predicciones_csv(resultado, ruta_predicciones)

    sin_interfaz = argumentos.no_gui or _entorno_sin_pantalla()
    _configurar_backend(sin_interfaz)
    from iris_classifier.visualization import generar_visualizaciones

    generar_visualizaciones(
        resultado,
        atributos,
        ruta_fronteras,
        ruta_metricas,
        mostrar_ventana=not sin_interfaz,
    )
    print("\nArchivos generados:")
    for ruta in (ruta_fronteras, ruta_metricas, ruta_csv, ruta_predicciones):
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
