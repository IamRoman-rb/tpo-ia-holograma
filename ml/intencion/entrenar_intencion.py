"""Entrena el intérprete de pedidos propio (motor "modelo" del backend): el respaldo del LLM sin internet.

Dos partes, las dos entrenadas con ml/intencion/frases_entrenamiento.csv (generar_frases.py):
  1. Acción: la frase pasa por un modelo de embeddings de oraciones (congelado) y un clasificador decide la
     acción (11 clases). Se comparan dos modelos de embeddings y cabezales de distinto tamaño, desde una
     regresión logística hasta una red de tres capas ocultas.
  2. Objeto: un etiquetador marca qué palabras de la frase nombran el objeto o la pieza ("mostrame la turbina
     de acero desde atrás" → "turbina de acero"). Es una regresión logística por palabra, con la palabra y sus
     vecinas como rasgos.
La vista y el factor de escala son conjuntos cerrados: los siguen sacando las reglas (backend/intent.py).

Se elige con VALIDACIÓN (eval/frases_validacion.csv, escritas a mano antes de entrenar) y se informa PRUEBA
(eval/frases_prueba.csv, las mismas 38 frases con las que se midieron las reglas y el LLM).
El resultado de punta a punta (acción + objeto + vista + escala, con los filtros de seguridad) lo da
eval/eval_intent.py --motor modelo, con el mismo código del backend.

Uso (desde la raíz):  backend/.venv/Scripts/python ml/intencion/entrenar_intencion.py
Salida: backend/modelos/intencion.json (pesos, sin pickle) y ml/resultados/intencion_reporte.json
"""
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
import intent_modelo  # noqa: E402  (rasgos del etiquetador y tokenización: los mismos que usa el backend)

SALIDA = RAIZ / "backend" / "modelos" / "intencion.json"
REPORTE = RAIZ / "ml" / "resultados" / "intencion_reporte.json"

EMBEDDINGS = ["paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small"]
CABEZALES = {
    "logistica": lambda: LogisticRegression(C=10, max_iter=3000),
    "mlp_256": lambda: MLPClassifier((256,), alpha=1e-3, max_iter=600, early_stopping=True, random_state=0),
    "mlp_512_256": lambda: MLPClassifier((512, 256), alpha=1e-3, max_iter=600, early_stopping=True, random_state=0),
    "mlp_1024_512_256": lambda: MLPClassifier((1024, 512, 256), alpha=1e-3, max_iter=600, early_stopping=True,
                                              random_state=0),
}
TAMANO = {"logistica": 0, "mlp_256": 1, "mlp_512_256": 2, "mlp_1024_512_256": 3}


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def codificar(modelo, frases, nombre):
    prefijo = intent_modelo.prefijo_consulta(nombre)
    return modelo.encode([prefijo + f for f in frases], normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False)


def capas(clf):
    """Pesos de cualquier cabezal como capas densas (la logística es una sola capa)."""
    if isinstance(clf, LogisticRegression):
        return [{"W": clf.coef_.T, "b": clf.intercept_}]
    return [{"W": W, "b": b} for W, b in zip(clf.coefs_, clf.intercepts_)]


# ---------------------------------------------------------------- etiquetador del objeto

def etiquetas_objeto(frase, objeto):
    """1 en las palabras de la frase que forman el objeto (primera aparición), 0 en las demás."""
    palabras = intent_modelo.palabras(frase)
    objetivo = intent_modelo.palabras(objeto) if objeto else []
    marcas = [0] * len(palabras)
    n = len(objetivo)
    for i in range(len(palabras) - n + 1):
        if n and palabras[i:i + n] == objetivo:
            marcas[i:i + n] = [1] * n
            break
    return palabras, marcas


def entrenar_etiquetador(filas):
    from sklearn.feature_extraction import DictVectorizer
    X, y = [], []
    for fila in filas:
        if fila["accion"] not in intent_modelo.ACCIONES_CON_OBJETO:
            continue
        palabras, marcas = etiquetas_objeto(fila["frase"], fila["objeto"])
        X += [intent_modelo.rasgos(palabras, i, fila["accion"]) for i in range(len(palabras))]
        y += marcas
    vec = DictVectorizer()
    clf = LogisticRegression(C=3, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(X), y)
    pesos = {nombre: round(float(w), 5) for nombre, w in zip(vec.get_feature_names_out(), clf.coef_[0])
             if abs(w) > 1e-4}
    return {"pesos": pesos, "sesgo": round(float(clf.intercept_[0]), 5)}


def main():
    from sentence_transformers import SentenceTransformer

    entrenamiento = leer(AQUI / "frases_entrenamiento.csv")
    validacion = leer(RAIZ / "eval" / "frases_validacion.csv")
    prueba = leer(RAIZ / "eval" / "frases_prueba.csv")
    acciones = sorted({f["accion"] for f in entrenamiento})
    y_ent = [f["accion"] for f in entrenamiento]
    y_val = [f["accion"] for f in validacion]
    y_pru = [f["accion"] for f in prueba]
    print(f"entrenamiento {len(entrenamiento)} · validación {len(validacion)} · prueba {len(prueba)} · "
          f"{len(acciones)} acciones")

    # 1) Acción: embeddings x cabezales, elegido con validación (a igual exactitud, el más chico)
    resultados, mejor = [], None
    for nombre in EMBEDDINGS:
        modelo = SentenceTransformer(nombre)
        E_ent = codificar(modelo, [f["frase"] for f in entrenamiento], nombre)
        E_val = codificar(modelo, [f["frase"] for f in validacion], nombre)
        E_pru = codificar(modelo, [f["frase"] for f in prueba], nombre)
        for cabezal, crear in CABEZALES.items():
            t0 = time.perf_counter()
            clf = crear().fit(E_ent, y_ent)
            p_val, p_pru = clf.predict(E_val), clf.predict(E_pru)
            fila = {"embeddings": nombre, "cabezal": cabezal,
                    "exactitud_validacion": round(accuracy_score(y_val, p_val), 4),
                    "f1_validacion": round(f1_score(y_val, p_val, average="macro"), 4),
                    "exactitud_prueba": round(accuracy_score(y_pru, p_pru), 4),
                    "segundos": round(time.perf_counter() - t0, 1)}
            resultados.append(fila)
            print(f"  {nombre.split('/')[-1]:38s} {cabezal:18s} validación {fila['exactitud_validacion']:.3f} "
                  f"(F1 {fila['f1_validacion']:.3f}) · prueba {fila['exactitud_prueba']:.3f} · {fila['segundos']} s")
            clave = (fila["exactitud_validacion"], fila["f1_validacion"], -TAMANO[cabezal])
            if mejor is None or clave > mejor[0]:
                mejor = (clave, nombre, cabezal, clf, p_val)

    _, nombre, cabezal, clf, p_val = mejor
    print(f"\nelegido: {nombre} + {cabezal}")
    for f, p in zip(validacion, p_val):
        if p != f["accion"]:
            print(f"  validación, falla: {f['frase']!r} → {p} (era {f['accion']})")

    # 2) Etiquetador del objeto (entrenado con las mismas frases)
    etiquetador = entrenar_etiquetador(entrenamiento)
    print(f"etiquetador: {len(etiquetador['pesos'])} rasgos con peso")

    # 3) Exportar: pesos en JSON (el backend hace la cuenta con numpy, sin pickle ni scikit-learn)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    datos = {
        "embeddings": nombre,
        "acciones": [str(c) for c in clf.classes_],
        "capas": [{"W": intent_modelo.empaquetar(c["W"]), "b": intent_modelo.empaquetar(c["b"])} for c in capas(clf)],
        "activacion_oculta": "relu",
        "etiquetador": etiquetador,
        "entrenamiento": {"frases": len(entrenamiento), "fecha": date.today().isoformat(),
                          "origen": "ml/intencion/generar_frases.py"},
    }
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"modelo -> {SALIDA.relative_to(RAIZ)} ({SALIDA.stat().st_size // 1024} KB)")

    REPORTE.parent.mkdir(exist_ok=True)
    REPORTE.write_text(json.dumps({
        "fecha": date.today().isoformat(),
        "frases": {"entrenamiento": len(entrenamiento), "validacion": len(validacion), "prueba": len(prueba)},
        "comparacion_accion": resultados,
        "elegido": {"embeddings": nombre, "cabezal": cabezal},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
