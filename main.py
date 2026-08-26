"""Punto de entrada de la demostración académica de Value Iteration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from gridworld.config import Configuracion
from gridworld.environment import GridWorld
from gridworld.value_iteration import IteracionDeValores

RUTA_RESULTADO = Path("output") / "resultado_final.png"


def crear_parser() -> argparse.ArgumentParser:
    """Construye la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Resuelve un GridWorld 15x15 mediante Value Iteration y muestra "
            "la política óptima."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para generar obstáculos aleatorios (predeterminado: 42).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Pausa en segundos entre cuadros de animación (predeterminado: 0.15).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="No abre ventanas; únicamente guarda la imagen final.",
    )
    parser.add_argument(
        "--show-iterations",
        action="store_true",
        help="Anima los valores y la política provisional durante la convergencia.",
    )
    parser.add_argument(
        "--random-obstacles",
        action="store_true",
        help="Sustituye los cinco obstáculos fijos por otros aleatorios resolubles.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Límite de iteraciones de Bellman (predeterminado: 1000).",
    )
    return parser


def entorno_sin_pantalla() -> bool:
    """Detecta servidores Unix sin DISPLAY; Windows y macOS se prueban normalmente."""
    if platform.system() in {"Windows", "Darwin"}:
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def configurar_backend(modo_sin_interfaz: bool) -> None:
    """Prepara una caché local y Agg cuando no existe una pantalla."""
    ruta_config = Path(".matplotlib-cache").resolve()
    ruta_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ruta_config))
    if modo_sin_interfaz:
        import matplotlib

        matplotlib.use("Agg", force=True)


def ejecutar(argumentos: argparse.Namespace) -> int:
    """Ejecuta planificación, reporte y visualización."""
    if argumentos.speed <= 0.0:
        raise ValueError("--speed debe ser mayor que cero.")
    if argumentos.max_iterations <= 0:
        raise ValueError("--max-iterations debe ser mayor que cero.")

    configuracion = Configuracion(max_iteraciones=argumentos.max_iterations)
    if argumentos.random_obstacles:
        configuracion = configuracion.con_obstaculos_aleatorios(argumentos.seed)
    entorno = GridWorld(configuracion)
    planificador = IteracionDeValores(entorno)
    resultado = planificador.resolver(mostrar_progreso_terminal=True)

    print("\n=== Resumen del experimento ===")
    print(f"Convergió: {'Sí' if resultado.convergio else 'No'}")
    print(f"Iteraciones: {resultado.iteraciones}")
    print(f"Obstáculos: {sorted(configuracion.obstaculos)}")
    print(f"Camino encontrado: {list(resultado.camino)}")
    print(f"Cantidad de pasos: {len(resultado.camino) - 1}")
    print(f"Recompensa acumulada: {resultado.recompensa_acumulada:.2f}")
    print(f"Tiempo de ejecución: {resultado.tiempo_segundos:.6f} s")

    modo_sin_interfaz = argumentos.no_gui or entorno_sin_pantalla()
    if entorno_sin_pantalla() and not argumentos.no_gui:
        print("Aviso: no se detectó una pantalla; se utilizará el modo sin interfaz.")
    configurar_backend(modo_sin_interfaz)
    from gridworld.visualization import VisualizadorGridWorld

    visualizador = VisualizadorGridWorld(entorno)
    try:
        visualizador.visualizar(
            resultado=resultado,
            ruta_salida=RUTA_RESULTADO,
            velocidad=argumentos.speed,
            mostrar_ventana=not modo_sin_interfaz,
            mostrar_iteraciones=argumentos.show_iterations,
        )
    except Exception as error_grafico:
        if modo_sin_interfaz:
            raise
        print(
            "Aviso: la interfaz gráfica no pudo abrirse; se guardará el "
            f"resultado con el backend Agg ({error_grafico})."
        )
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.switch_backend("Agg")
        visualizador.visualizar(
            resultado=resultado,
            ruta_salida=RUTA_RESULTADO,
            velocidad=argumentos.speed,
            mostrar_ventana=False,
            mostrar_iteraciones=False,
        )

    print(f"Imagen final: {RUTA_RESULTADO.resolve()}")
    return 0


def main() -> int:
    """Analiza argumentos y convierte errores esperables en mensajes claros."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = crear_parser()
    argumentos = parser.parse_args()
    try:
        return ejecutar(argumentos)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
