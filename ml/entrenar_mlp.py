"""VERSIÓN 1 del MLP de gestos (scikit-learn, muestra de 30k de HaGRID). La red del visor ahora sale de
ml/entrenar_red_gestos.py (versión 2: PyTorch, 250k fotos, aumento de datos). Este script queda para reproducir
la versión 1: lee los landmarks de ml/data/externo/v1_30k/ y necesita las reglas evaluadas sobre ESA prueba.

Entrena el MLP de gestos con los landmarks de HaGRID (ml/extraer_hagrid.py) y lo compara con las reglas.

- Partición fija por persona (preparar.particion): se elige la arquitectura con VALIDACIÓN y se informa
  el resultado en PRUEBA, con personas que el modelo nunca vio (sección 15: persona no vista).
- Aumento de datos: cada mano de entrenamiento se agrega también espejada (x → −x), para que el modelo
  funcione igual con la mano izquierda y la derecha.
- Clases desbalanceadas ("ninguno" es el 63 %): pesos por clase inversos a su frecuencia.
- Las reglas se evalúan sobre las MISMAS filas de prueba con el código del visor (ml/evaluar_reglas.mjs).
- Exporta los pesos a JSON para el navegador (frontend/js/mlp.js hace la inferencia) y casos de prueba
  para verificar que JS y Python dan lo mismo.

Uso (desde la raíz):
    node ml/evaluar_reglas.mjs ml/data/externo/prueba ml/resultados/hagrid_reglas_prueba.csv
    ml/.venv-mediapipe/Scripts/python.exe ml/entrenar_mlp.py
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.stdout.reconfigure(encoding="utf-8")
ML = Path(__file__).resolve().parent
RAIZ = ML.parent
sys.path.insert(0, str(ML))
import preparar  # noqa: E402

DATOS = ML / "data" / "externo" / "v1_30k"
RESULTADOS = ML / "resultados"
MODELO_WEB = RAIZ / "frontend" / "ml" / "mlp_gestos.json"
CASOS_TEST = RAIZ / "frontend" / "tests" / "datos" / "mlp_casos.json"
ACCIONES = [c for c in preparar.CLASES if c != "ninguno"]

CANDIDATOS = [
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-3},
]


def espejar(X):
    """Mano espejada: invierte x en los 21 puntos (columnas 0, 3, 6, …)."""
    Y = X.copy()
    Y[:, 0::3] *= -1
    return Y


def entrenar(params, X, y):
    modelo = MLPClassifier(activation="relu", solver="adam", max_iter=400, early_stopping=True,
                           validation_fraction=0.1, n_iter_no_change=15, random_state=0, **params)
    modelo.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return modelo


def falsas_alarmas(y_real, y_pred):
    """Proporción de manos SIN gesto que el clasificador toma por un gesto: cada una es una acción no pedida."""
    ninguno = np.asarray(y_real) == "ninguno"
    return float(np.mean(np.asarray(y_pred)[ninguno] != "ninguno"))


def resumen(nombre, y_real, y_pred):
    return {
        "metodo": nombre,
        "exactitud": round(float(np.mean(np.asarray(y_real) == np.asarray(y_pred))), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro")), 4),
        "recall_gestos": round(float(np.mean([np.mean(np.asarray(y_pred)[np.asarray(y_real) == c] == c)
                                              for c in ACCIONES])), 4),
        "falsas_alarmas_ninguno": round(falsas_alarmas(y_real, y_pred), 4),
    }


def main():
    df = preparar.cargar(DATOS)
    df["particion"] = df["persona"].map(preparar.particion)
    X = preparar.normalizar(df)
    y = df["clase"].to_numpy()
    parte = df["particion"].to_numpy()
    ent, val, pru = (parte == "entrenamiento"), (parte == "validacion"), (parte == "prueba")
    X_ent = np.vstack([X[ent], espejar(X[ent])])
    y_ent = np.concatenate([y[ent], y[ent]])
    print(f"entrenamiento {ent.sum()} (×2 con espejo) · validación {val.sum()} · prueba {pru.sum()}")

    # 1) Elegir la arquitectura con validación (F1 macro: todas las clases pesan igual)
    elegido, mejor = None, -1
    for params in CANDIDATOS:
        t0 = time.perf_counter()
        modelo = entrenar(params, X_ent, y_ent)
        f1 = f1_score(y[val], modelo.predict(X[val]), average="macro")
        print(f"  {params} → F1 validación {f1:.3f} ({time.perf_counter() - t0:.0f} s, {modelo.n_iter_} épocas)")
        if f1 > mejor:
            elegido, mejor, final = params, f1, modelo

    # 2) Umbral de confianza: por debajo, la mano cuenta como "ninguno" (menos acciones no pedidas).
    #    También se elige con validación, buscando el mejor F1 macro.
    proba_val = final.predict_proba(X[val])
    umbral, mejor_u = 0.0, -1
    for u in np.arange(0.3, 0.96, 0.05):
        pred = np.where(proba_val.max(1) >= u, final.classes_[proba_val.argmax(1)], "ninguno")
        f1 = f1_score(y[val], pred, average="macro")
        if f1 > mejor_u + 1e-4:
            umbral, mejor_u = float(round(u, 2)), f1
    print(f"elegido {elegido} · umbral de confianza {umbral} (F1 validación {mejor_u:.3f})")

    # 3) Prueba: personas no vistas. MLP vs reglas sobre las mismas filas.
    proba = final.predict_proba(X[pru])
    pred_mlp = np.where(proba.max(1) >= umbral, final.classes_[proba.argmax(1)], "ninguno")
    reglas = pd.read_csv(RESULTADOS / "hagrid_reglas_prueba.csv")["prediccion"].to_numpy()
    if len(reglas) != pru.sum():
        sys.exit("Las predicciones de las reglas no corresponden a la partición de prueba: correr evaluar_reglas.mjs")
    y_pru = y[pru]
    comparacion = [resumen("reglas", y_pru, reglas), resumen("mlp", y_pru, pred_mlp)]
    print("\n" + pd.DataFrame(comparacion).to_string(index=False))
    print("\nMLP por clase (prueba):\n" + classification_report(y_pru, pred_mlp, digits=3, zero_division=0))
    print("Reglas por clase (prueba):\n" + classification_report(y_pru, reglas, digits=3, zero_division=0))

    clases = list(final.classes_)
    RESULTADOS.mkdir(exist_ok=True)
    pd.DataFrame(confusion_matrix(y_pru, pred_mlp, labels=clases), index=clases, columns=clases) \
        .to_csv(RESULTADOS / "hagrid_mlp_confusion.csv", encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_pru, reglas, labels=clases), index=clases, columns=clases) \
        .to_csv(RESULTADOS / "hagrid_reglas_confusion.csv", encoding="utf-8")
    reporte = {
        "fecha": date.today().isoformat(),
        "dataset": "HaGRID sample 30k 384p (Hugging Face cj-mills/hagrid-sample-30k-384p), CC BY-SA 4.0",
        "filas": {"entrenamiento": int(ent.sum()), "validacion": int(val.sum()), "prueba": int(pru.sum())},
        "personas": {k: int(df.loc[parte == k, "persona"].nunique()) for k in ("entrenamiento", "validacion", "prueba")},
        "arquitectura": {"capas_ocultas": list(elegido["hidden_layer_sizes"]), "alpha": elegido["alpha"]},
        "umbral_confianza": umbral,
        "comparacion_prueba": comparacion,
        "por_clase_mlp": classification_report(y_pru, pred_mlp, output_dict=True, zero_division=0),
        "por_clase_reglas": classification_report(y_pru, reglas, output_dict=True, zero_division=0),
    }
    (RESULTADOS / "hagrid_mlp_reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Exportar para el navegador
    MODELO_WEB.parent.mkdir(parents=True, exist_ok=True)
    web = {
        "clases": clases,
        "umbral_confianza": umbral,
        "capas": [{"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()}
                  for W, b in zip(final.coefs_, final.intercepts_)],
        "activacion": "relu",
        "entrada": "63 valores normalizados (frontend/js/normalize.js: aVector(normalizar(puntos, aspecto)))",
        "origen": reporte["dataset"],
        "fecha": reporte["fecha"],
    }
    MODELO_WEB.write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rng = np.random.default_rng(0)
    muestras = rng.choice(np.flatnonzero(pru), 12, replace=False)
    casos = [{"x": np.round(X[i], 6).tolist(), "proba": np.round(final.predict_proba(X[i:i + 1])[0], 5).tolist()}
             for i in muestras]
    CASOS_TEST.parent.mkdir(parents=True, exist_ok=True)
    CASOS_TEST.write_text(json.dumps({"clases": clases, "casos": casos}), encoding="utf-8")
    print(f"\nModelo para el navegador: {MODELO_WEB.relative_to(RAIZ)} ({MODELO_WEB.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
