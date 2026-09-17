"""Compara el MLP anterior de Wine con dos capas ocultas y selecciona su LR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

from ann_wine import (
    _guardar_csv, _metricas, crear_modelo, describir_red,
    ejecutar_experimento, parametros_red,
)


def crear_mlp(learning_rate=0.01, max_iter=2000, seed=42):
    """Solo cambia la arquitectura del ejercicio previo; conserva su escalado."""
    return crear_modelo(learning_rate, 16, max_iter, seed).set_params(
        ann__hidden_layer_sizes=(16, 8),
    )


def ejecutar_comparacion(learning_rates=(0.001, 0.01, 0.1), max_iter=2000,
                        cv_folds=5, seed=42):
    """Reentrena ambas arquitecturas con idénticos índices y folds de CV.

    La arquitectura nueva se fija antes de evaluar. Cada arquitectura escoge
    su LR por F1 macro de CV sobre train, nunca por su desempeño en test.
    """
    anterior = ejecutar_experimento(
        learning_rates=learning_rates, hidden_units=16, max_iter=max_iter,
        cv_folds=cv_folds, seed=seed,
    )
    x_train, y_train = anterior["x_train"], anterior["y_train"]
    x_test, y_test = anterior["x_test"], anterior["y_test"]
    resumen, detalles, modelos, curvas = [], [], {}, {}
    # El ejercicio previo valida opciones y ordena/deduplica las tasas.
    for fila_anterior in anterior["cv_summary"]:
        tasa = fila_anterior["learning_rate"]
        modelo = crear_mlp(tasa, max_iter, seed)
        cv = cross_validate(
            modelo, x_train, y_train, cv=anterior["folds"],
            scoring={"accuracy": "accuracy", "f1": "f1_macro", "loss": "neg_log_loss"},
            return_train_score=True, return_estimator=True, error_score="raise",
        )
        epocas = [estimador.named_steps["ann"].n_iter_ for estimador in cv["estimator"]]
        resumen.append({
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
        for i, epoca in enumerate(epocas):
            detalles.append({
                "learning_rate": tasa, "fold": i + 1,
                "train_accuracy": float(cv["train_accuracy"][i]),
                "validation_accuracy": float(cv["test_accuracy"][i]),
                "validation_f1_macro": float(cv["test_f1"][i]),
                "validation_log_loss": float(-cv["test_loss"][i]), "epochs": epoca,
            })
        # Estas curvas usan todo train; no son curvas del conjunto de prueba.
        modelo.fit(x_train, y_train)
        modelos[tasa] = modelo
        curvas[tasa] = modelo.named_steps["ann"].loss_curve_

    mejor_lr = max(resumen, key=lambda fila: fila["f1_mean"])["learning_rate"]
    modelo = modelos[mejor_lr]
    modelo_2d = crear_mlp(mejor_lr, max_iter, seed)
    modelo_2d.fit(x_train[:, [0, 6]], y_train)
    predicciones = modelo.predict(x_test)
    nuevo = {
        clave: anterior[clave] for clave in (
            "x_train", "x_test", "y_train", "y_test", "idx_train", "idx_test",
            "folds", "feature_names", "seed",
        )
    }
    nuevo.update({
        "model": modelo, "model_2d": modelo_2d,
        "cv_summary": resumen, "cv_details": detalles,
        "best_lr": mejor_lr, "loss_curves": curvas,
        "metrics": {
            "entrenamiento": _metricas(modelo, x_train, y_train),
            "prueba": _metricas(modelo, x_test, y_test),
            "frontera_2d": _metricas(modelo_2d, x_test[:, [0, 6]], y_test),
        },
        "confusion": confusion_matrix(y_test, predicciones, labels=[0, 1, 2]),
        "predictions": predicciones, "probabilities": modelo.predict_proba(x_test),
        "report": classification_report(
            y_test, predicciones, target_names=["Cultivar 1", "Cultivar 2", "Cultivar 3"],
            zero_division=0,
        ),
    })
    return {"anterior": anterior, "mlp": nuevo}


def exportar_comparacion(resultado, directorio):
    """Exporta evaluación, selección de LR y parámetros de ambas redes."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    tablas = {nombre: [] for nombre in (
        "metricas", "cv", "learning_rates", "confusion", "costos",
        "predicciones", "parametros", "escalado",
    )}
    for nombre, datos in resultado.items():
        tablas["metricas"].extend(
            {"modelo": nombre, "conjunto": conjunto,
             "learning_rate": datos["best_lr"], **metricas}
            for conjunto, metricas in datos["metrics"].items()
        )
        for tabla, clave in (("cv", "cv_details"), ("learning_rates", "cv_summary")):
            tablas[tabla].extend({"modelo": nombre, **fila} for fila in datos[clave])
        tablas["confusion"].extend(
            {"modelo": nombre, "clase_real": real, "clase_predicha": predicha,
             "cantidad": int(datos["confusion"][real, predicha])}
            for real in range(3) for predicha in range(3)
        )
        tablas["costos"].extend(
            {"modelo": nombre, "learning_rate": tasa, "epoca": i + 1,
             "costo_entrenamiento": float(costo)}
            for tasa, costos in datos["loss_curves"].items() for i, costo in enumerate(costos)
        )
        tablas["predicciones"].extend(
            {"modelo": nombre, "indice_original": int(indice),
             "clase_real": int(real), "clase_predicha": int(predicha),
             **{f"probabilidad_clase_{c}": float(p) for c, p in enumerate(probabilidades)}}
            for indice, real, predicha, probabilidades in zip(
                datos["idx_test"], datos["y_test"], datos["predictions"], datos["probabilities"],
            )
        )
        tablas["parametros"].extend({"modelo": nombre, **fila} for fila in parametros_red(datos))
        escalador = datos["model"].named_steps["escalador"]
        tablas["escalado"].extend(
            {"modelo": nombre, "atributo": atributo, "media": float(media), "escala": float(escala)}
            for atributo, media, escala in zip(
                datos["feature_names"], escalador.mean_, escalador.scale_,
            )
        )
    for nombre, filas in tablas.items():
        _guardar_csv(directorio / f"mlp_wine_{nombre}.csv", filas)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    try:
        resultado = ejecutar_comparacion(
            args.learning_rates, args.max_iter, args.cv_folds, args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print("Wine: mismos 124 ejemplos de entrenamiento / 54 de prueba (70/30).")
    print("La ANN anterior YA es un MLP; se compara una frente a dos capas ocultas.")
    print(f"Learning rate seleccionado por F1 macro de CV ({args.cv_folds} folds solo en train).")
    for nombre, datos in resultado.items():
        red = describir_red(datos)
        print(f"\n{nombre.upper()}: " + " -> ".join(map(str, red["dimensiones"])))
        print(f"{red['parametros']} parámetros; LR={datos['best_lr']:g}; "
              f"{red['epocas']} épocas; {red['actualizaciones']} actualizaciones Adam.")
        for fila in datos["cv_summary"]:
            print(f"  LR={fila['learning_rate']:g}: "
                  f"F1 CV={fila['f1_mean']:.4f} ± {fila['f1_std']:.4f}; "
                  f"accuracy CV={fila['accuracy_mean']:.4f}; "
                  f"épocas medias={fila['epochs_mean']:.1f}; "
                  f"folds al límite={fila['folds_at_limit']}")
        for conjunto in ("entrenamiento", "prueba"):
            m = datos["metrics"][conjunto]
            print(f"  {conjunto}: accuracy={m['accuracy']:.4f}; "
                  f"precisión={m['precision_macro']:.4f}; recall={m['recall_macro']:.4f}; "
                  f"F1={m['f1_macro']:.4f}; error={m['error']:.4f}; log-loss={m['log_loss']:.4f}")
        print("  Matriz de confusión: filas reales / columnas predichas (clases 0, 1, 2)")
        print(datos["confusion"])
        print(datos["report"])
    diferencia = (resultado["mlp"]["metrics"]["prueba"]["accuracy"]
                  - resultado["anterior"]["metrics"]["prueba"]["accuracy"]) * 100
    print(f"Diferencia de accuracy de prueba (MLP nuevo - anterior): {diferencia:+.2f} puntos.")
    print("Más capas no garantizan mejor desempeño. Comparación exploratoria en un test ya utilizado.")
    exportar_comparacion(resultado, args.output_dir)
    cache = Path(".matplotlib-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    sin_pantalla = os.name != "nt" and sys.platform != "darwin" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if args.no_gui or sin_pantalla:
        matplotlib.use("Agg")
    from mlp_wine_visualization import generar_visualizaciones
    generar_visualizaciones(resultado, args.output_dir, not (args.no_gui or sin_pantalla))
    print(f"Resultados guardados en: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
