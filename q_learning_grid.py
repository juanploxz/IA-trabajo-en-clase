"""CLI para entrenar Q-Learning y compararlo con Value Iteration."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import platform
import sys

from gridworld.config import Configuracion
from gridworld.environment import GridWorld
from gridworld.q_learning import (
    AgenteQLearning,
    HiperparametrosQLearning,
    ResultadoQLearning,
)
from gridworld.value_iteration import IteracionDeValores, ResultadoValueIteration

RUTA_GRAFICA = Path("output") / "comparacion_q_learning.png"
RUTA_CSV = Path("output") / "comparacion_q_learning.csv"
RUTA_MODELO = Path("models") / "q_learning_grid.npz"


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena Q-Learning en el GridWorld 15x15 y compara sus resultados "
            "con Value Iteration."
        )
    )
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--random-obstacles", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--output", type=Path, default=RUTA_GRAFICA)
    parser.add_argument("--csv", type=Path, default=RUTA_CSV)
    parser.add_argument("--model", type=Path, default=RUTA_MODELO)
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


def _escribir_csv(
    ruta: Path,
    value_iteration: ResultadoValueIteration,
    q_learning: ResultadoQLearning,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pasos_q = len(q_learning.camino) - 1 if q_learning.camino else "No llega"
    filas = [
        ("enfoque", "Planificación con modelo", "Aprendizaje sin modelo"),
        ("actualizaciones", value_iteration.iteraciones, q_learning.episodios),
        ("interacciones_reales", 0, q_learning.total_interacciones),
        (
            "tiempo_segundos",
            value_iteration.tiempo_segundos,
            q_learning.tiempo_segundos,
        ),
        ("pasos_camino", len(value_iteration.camino) - 1, pasos_q),
        (
            "recompensa_camino",
            value_iteration.recompensa_acumulada,
            q_learning.recompensa_acumulada,
        ),
        ("tasa_exito_final", 1.0, q_learning.tasa_exito_final),
        (
            "estados_no_terminales_actualizados",
            "219 por modelo",
            q_learning.estados_visitados,
        ),
        ("episodio_estable_95", "No aplica", q_learning.episodio_estable),
    ]
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("metrica", "value_iteration", "q_learning"))
        escritor.writerows(filas)


def _imprimir_comparacion(
    value_iteration: ResultadoValueIteration,
    q_learning: ResultadoQLearning,
) -> None:
    pasos_vi = len(value_iteration.camino) - 1
    pasos_q = len(q_learning.camino) - 1 if q_learning.camino else None
    print("\n=== Comparación experimental ===")
    print(f"{'Métrica':<30}{'Value Iteration':>20}{'Q-Learning':>20}")
    print("-" * 70)
    print(f"{'Conoce el modelo':<30}{'Sí':>20}{'No':>20}")
    print(
        f"{'Iteraciones / episodios':<30}"
        f"{value_iteration.iteraciones:>20}{q_learning.episodios:>20}"
    )
    print(
        f"{'Interacciones reales':<30}{0:>20}"
        f"{q_learning.total_interacciones:>20}"
    )
    print(
        f"{'Tiempo (s)':<30}{value_iteration.tiempo_segundos:>20.6f}"
        f"{q_learning.tiempo_segundos:>20.6f}"
    )
    print(
        f"{'Pasos de la ruta':<30}{pasos_vi:>20}"
        f"{str(pasos_q if pasos_q is not None else 'No llega'):>20}"
    )
    print(
        f"{'Recompensa de la ruta':<30}"
        f"{value_iteration.recompensa_acumulada:>20.2f}"
        f"{q_learning.recompensa_acumulada:>20.2f}"
    )
    print(
        f"{'Éxito últimos 100':<30}{'100.0%':>20}"
        f"{q_learning.tasa_exito_final:>19.1%}"
    )
    print(
        f"{'Estados actualizados':<30}{'219 (modelo)':>20}"
        f"{q_learning.estados_visitados:>20}"
    )
    episodio_estable = q_learning.episodio_estable or "No alcanzado"
    print(
        f"{'Primer episodio con 95%':<30}{'No aplica':>20}"
        f"{str(episodio_estable):>20}"
    )
    coincide = pasos_q == pasos_vi and (
        q_learning.recompensa_acumulada
        == value_iteration.recompensa_acumulada
    )
    print(f"\nQ-Learning encontró la ruta óptima: {'Sí' if coincide else 'No'}")
    print(f"Camino Q-Learning: {list(q_learning.camino)}")


def ejecutar(argumentos: argparse.Namespace) -> int:
    configuracion = Configuracion(gamma=argumentos.gamma)
    if argumentos.random_obstacles:
        configuracion = configuracion.con_obstaculos_aleatorios(argumentos.seed)
    entorno = GridWorld(configuracion)
    referencia = IteracionDeValores(entorno).resolver(
        mostrar_progreso_terminal=False
    )
    parametros = HiperparametrosQLearning(
        episodios=argumentos.episodes,
        alpha=argumentos.alpha,
        gamma=argumentos.gamma,
        epsilon_inicial=argumentos.epsilon_start,
        epsilon_minimo=argumentos.epsilon_min,
        decaimiento_epsilon=argumentos.epsilon_decay,
        max_pasos=argumentos.max_steps,
        semilla=argumentos.seed,
    )
    agente = AgenteQLearning(entorno, parametros)
    resultado = agente.entrenar(
        mostrar_progreso=True,
        intervalo_reporte=argumentos.report_every,
    )
    if not resultado.politica_llega_meta:
        raise RuntimeError(
            "La política final no llega a la meta. Aumente --episodes o ajuste epsilon."
        )
    agente.guardar(argumentos.model, resultado)
    _imprimir_comparacion(referencia, resultado)
    _escribir_csv(argumentos.csv, referencia, resultado)

    sin_interfaz = argumentos.no_gui or _entorno_sin_pantalla()
    _configurar_backend(sin_interfaz)
    from gridworld.q_learning_comparison import crear_figura_comparativa

    crear_figura_comparativa(
        entorno,
        resultado,
        referencia,
        argumentos.output,
        mostrar_ventana=not sin_interfaz,
    )
    print(f"Modelo: {argumentos.model.resolve()}")
    print(f"CSV: {argumentos.csv.resolve()}")
    print(f"Gráfica: {argumentos.output.resolve()}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argumentos = crear_parser().parse_args()
    try:
        return ejecutar(argumentos)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
