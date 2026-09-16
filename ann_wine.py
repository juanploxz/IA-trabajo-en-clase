"""ANN con sigmoide para Wine: split 70/30 y selección de learning rate por CV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crear_modelo(learning_rate=0.01, hidden_units=16, max_iter=2000, seed=42):
    """El escalado se aprende dentro de cada fold; sigmoide oculta y softmax final."""
    return Pipeline([
        ("escalador", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(hidden_units,), activation="logistic",
            solver="adam", learning_rate_init=learning_rate,
            alpha=0.0001, batch_size=16, max_iter=max_iter,
            tol=1e-4, n_iter_no_change=20, early_stopping=False,
            random_state=seed,
        )),
    ])


def _metricas(modelo, x, y):
    predicciones = modelo.predict(x)
    exactitud = float(accuracy_score(y, predicciones))
    return {
        "accuracy": exactitud,
        "precision_macro": float(precision_score(y, predicciones, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predicciones, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predicciones, average="macro", zero_division=0)),
        "error": 1.0 - exactitud,
        "log_loss": float(log_loss(y, modelo.predict_proba(x), labels=[0, 1, 2])),
    }


def ejecutar_experimento(learning_rates=(0.001, 0.01, 0.1), hidden_units=16,
                        max_iter=2000, cv_folds=5, seed=42):
    """Selecciona tasa por F1 de CV y evalúa solo la seleccionada en prueba."""
    tasas = tuple(sorted(set(learning_rates)))
    if not tasas or any(not np.isfinite(tasa) or tasa <= 0 for tasa in tasas):
        raise ValueError("Las tasas de aprendizaje deben ser positivas y finitas.")
    if hidden_units < 1 or max_iter < 1 or cv_folds < 2 or seed < 0:
        raise ValueError("Use neuronas/épocas positivas, al menos 2 folds y semilla no negativa.")
    wine = load_wine()
    idx_train, idx_test = train_test_split(
        np.arange(len(wine.target)), test_size=0.30,
        stratify=wine.target, random_state=seed,
    )
    x_train, x_test = wine.data[idx_train], wine.data[idx_test]
    y_train, y_test = wine.target[idx_train], wine.target[idx_test]
    if cv_folds > np.bincount(y_train).min():
        raise ValueError("Hay más folds que muestras de la clase minoritaria en entrenamiento.")
    # Reutilizar exactamente los mismos folds permite comparar las tasas.
    folds = list(StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=seed,
    ).split(x_train, y_train))
    cv_summary, cv_detalle = [], []
    modelos, loss_curves = {}, {}
    for tasa in tasas:
        modelo = crear_modelo(tasa, hidden_units, max_iter, seed)
        cv = cross_validate(
            modelo, x_train, y_train, cv=folds,
            scoring={"accuracy": "accuracy", "f1": "f1_macro", "loss": "neg_log_loss"},
            return_train_score=True, return_estimator=True, error_score="raise",
        )
        epocas = [estimador.named_steps["ann"].n_iter_ for estimador in cv["estimator"]]
        cv_summary.append({
            "learning_rate": tasa,
            "accuracy_mean": float(np.mean(cv["test_accuracy"])),
            "accuracy_std": float(np.std(cv["test_accuracy"], ddof=1)),
            "f1_mean": float(np.mean(cv["test_f1"])),
            "f1_std": float(np.std(cv["test_f1"], ddof=1)),
            "train_accuracy_mean": float(np.mean(cv["train_accuracy"])),
            "log_loss_mean": float(-np.mean(cv["test_loss"])),
            "epochs_mean": float(np.mean(epocas)),
            "folds_at_limit": int(np.sum(np.asarray(epocas) >= max_iter)),
        })
        for i, estimador in enumerate(cv["estimator"]):
            cv_detalle.append({
                "learning_rate": tasa, "fold": i + 1,
                "train_accuracy": float(cv["train_accuracy"][i]),
                "validation_accuracy": float(cv["test_accuracy"][i]),
                "validation_f1_macro": float(cv["test_f1"][i]),
                "validation_log_loss": float(-cv["test_loss"][i]),
                "epochs": estimador.named_steps["ann"].n_iter_,
            })
        # Estas curvas usan exclusivamente entrenamiento; no se optimiza contra prueba.
        modelo.fit(x_train, y_train)
        modelos[tasa] = modelo
        loss_curves[tasa] = modelo.named_steps["ann"].loss_curve_

    # sorted(tasas) hace que un empate exacto favorezca la tasa más pequeña.
    mejor = max(cv_summary, key=lambda fila: fila["f1_mean"])
    best_lr = mejor["learning_rate"]
    modelo = modelos[best_lr]
    modelo_2d = crear_modelo(best_lr, hidden_units, max_iter, seed)
    modelo_2d.fit(x_train[:, [0, 6]], y_train)
    predicciones = modelo.predict(x_test)
    return {
        "x_train": x_train, "x_test": x_test,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test, "folds": folds,
        "model": modelo, "model_2d": modelo_2d,
        "cv_summary": cv_summary, "cv_details": cv_detalle,
        "best_lr": best_lr, "loss_curves": loss_curves,
        "metrics": {
            "entrenamiento": _metricas(modelo, x_train, y_train),
            "prueba": _metricas(modelo, x_test, y_test),
            "frontera_2d": _metricas(modelo_2d, x_test[:, [0, 6]], y_test),
        },
        "confusion": confusion_matrix(y_test, predicciones, labels=[0, 1, 2]),
        "predictions": predicciones, "probabilities": modelo.predict_proba(x_test),
        "seed": seed, "hidden_units": hidden_units,
        "report": classification_report(
            y_test, predicciones, target_names=["Cultivar 1", "Cultivar 2", "Cultivar 3"],
            zero_division=0,
        ),
    }


def _guardar_csv(ruta, filas):
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def exportar_resultados(resultado, directorio):
    """Guarda métricas, folds, selección de LR, predicciones y costos por época."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    _guardar_csv(directorio / "ann_wine_metricas.csv", [
        {"conjunto": nombre, "learning_rate": resultado["best_lr"], **metricas}
        for nombre, metricas in resultado["metrics"].items()
    ])
    _guardar_csv(directorio / "ann_wine_learning_rates.csv", resultado["cv_summary"])
    _guardar_csv(directorio / "ann_wine_cv.csv", resultado["cv_details"])
    _guardar_csv(directorio / "ann_wine_costos.csv", [
        {"learning_rate": tasa, "epoca": i + 1, "costo_entrenamiento": costo}
        for tasa, costos in resultado["loss_curves"].items()
        for i, costo in enumerate(costos)
    ])
    _guardar_csv(directorio / "ann_wine_predicciones.csv", [
        {"indice_original": int(indice), "clase_real": int(real),
         "clase_predicha": int(prediccion),
         **{f"probabilidad_clase_{c}": float(p) for c, p in enumerate(probabilidades)}}
        for indice, real, prediccion, probabilidades in zip(
            resultado["idx_test"], resultado["y_test"],
            resultado["predictions"], resultado["probabilities"],
        )
    ])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    try:
        resultado = ejecutar_experimento(
            args.learning_rates, args.hidden_units, args.max_iter, args.cv_folds, args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print("Wine: 124 entrenamiento / 54 prueba (70/30 estratificado)")
    print(f"CV de {args.cv_folds} folds SOLO en entrenamiento; selección por F1 macro.")
    for fila in resultado["cv_summary"]:
        print(f"LR={fila['learning_rate']:g}: accuracy CV={fila['accuracy_mean']:.4f}; "
              f"F1 CV={fila['f1_mean']:.4f} ± {fila['f1_std']:.4f}; "
              f"épocas promedio={fila['epochs_mean']:.0f}")
    print(f"\nLearning rate seleccionado: {resultado['best_lr']:g}")
    for nombre, metricas in resultado["metrics"].items():
        print(f"{nombre}: accuracy={metricas['accuracy']:.4f}; "
              f"F1={metricas['f1_macro']:.4f}; error={metricas['error']:.4f}; "
              f"log-loss={metricas['log_loss']:.4f}")
    print("\nMatriz de confusión (filas reales, columnas predichas):")
    print(resultado["confusion"])
    print(resultado["report"])
    exportar_resultados(resultado, args.output_dir)
    cache = Path(".matplotlib-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    sin_pantalla = os.name != "nt" and sys.platform != "darwin" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if args.no_gui or sin_pantalla:
        matplotlib.use("Agg")
    from ann_wine_visualization import generar_visualizaciones
    generar_visualizaciones(resultado, args.output_dir, not (args.no_gui or sin_pantalla))
    print(f"Resultados guardados en: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
