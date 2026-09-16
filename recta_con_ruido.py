"""Genera y dibuja 100 puntos alrededor de y = m*x + b con ruido normal."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def generar_muestras(
    m: float = 2.0,
    b: float = 1.0,
    ruido: float = 2.0,
    semilla: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve x, y observado, y ideal y ruido para exactamente 100 puntos."""
    if not np.all(np.isfinite([m, b, ruido])):
        raise ValueError("m, b y ruido deben ser números finitos.")
    if ruido < 0:
        raise ValueError("La desviación del ruido debe ser mayor o igual a cero.")
    if semilla < 0:
        raise ValueError("La semilla debe ser mayor o igual a cero.")

    generador = np.random.default_rng(semilla)
    x = generador.uniform(0.0, 10.0, size=100)
    dispersion = generador.normal(loc=0.0, scale=ruido, size=100)
    y_ideal = m * x + b
    y = y_ideal + dispersion
    return x, y, y_ideal, dispersion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=float, default=2.0, help="Pendiente; default: 2.")
    parser.add_argument("--b", type=float, default=1.0, help="Intercepto; default: 1.")
    parser.add_argument(
        "--noise", type=float, default=2.0,
        help="Desviación estándar del ruido; default: 2. Use 0 para quitarlo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla; default: 42.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-gui", action="store_true", help="Guardar sin abrir ventana.")
    args = parser.parse_args()

    try:
        x, y, y_ideal, ruido = generar_muestras(args.m, args.b, args.noise, args.seed)
    except ValueError as error:
        parser.error(str(error))

    # La semilla fija permite repetir los mismos puntos en clase.
    cache = Path(".matplotlib-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    if args.no_gui:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, eje = plt.subplots(figsize=(9, 6), layout="constrained")
    eje.scatter(x, y, s=40, alpha=0.8, color="#1565c0", label="100 puntos con ruido")
    # Solo la recta de referencia une puntos; las muestras son una dispersión.
    x_recta = np.array([0.0, 10.0])
    eje.plot(
        x_recta, args.m * x_recta + args.b,
        color="#c43824", linestyle="--", linewidth=2,
        label=f"Recta ideal: y = {args.m:g}x {args.b:+g}",
    )
    eje.set(
        xlabel="x", ylabel="y",
        title=f"100 muestras de una recta con ruido (desviación = {args.noise:g})",
    )
    eje.grid(alpha=0.2)
    eje.legend()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ruta_png = args.output_dir / "recta_con_ruido.png"
    ruta_csv = args.output_dir / "recta_con_ruido.csv"
    figura.savefig(ruta_png, dpi=160)
    np.savetxt(
        ruta_csv, np.column_stack((x, y, y_ideal, ruido)),
        delimiter=",", header="x,y,y_sin_ruido,ruido", comments="",
    )
    print(f"Generadas {len(x)} muestras: y = {args.m:g}x {args.b:+g} + ruido.")
    print(f"Gráfica: {ruta_png.resolve()}")
    print(f"Coordenadas: {ruta_csv.resolve()}")
    if not args.no_gui:
        plt.show()
    plt.close(figura)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
