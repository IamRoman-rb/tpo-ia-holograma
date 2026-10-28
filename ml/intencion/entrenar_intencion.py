"""Entrena el intérprete de pedidos propio (motor "modelo" del backend): el respaldo del LLM sin internet.

Dos partes, entrenadas con ml/intencion/frases_entrenamiento.csv (generar_frases.py, ~19 mil frases con unos
10 mil objetos distintos):
  1. Acción: la frase pasa por un modelo de embeddings de oraciones (congelado) y un clasificador decide la
     acción (11 clases). Se comparan dos modelos de embeddings y cabezales de distinto tamaño.
  2. Objeto: un etiquetador marca qué palabras de la frase nombran el objeto o la pieza ("mostrame la turbina
     de acero desde atrás" → "turbina de acero"). Se comparan:
       - "rasgos": regresión logística con la palabra, sus vecinas y su terminación (la versión anterior);
       - "contexto": el vector de cada palabra que da el modelo de embeddings (ya mira toda la frase), con el de
         sus vecinas, en una regresión logística o en una red con una capa oculta;
       - "combinado": el promedio de las probabilidades de los dos anteriores.
     A igual exactitud (diferencia menor a 0,2 puntos) se queda el más liviano.
La vista y el factor de escala son conjuntos cerrados: los siguen sacando las reglas (backend/intent.py).

Se elige con VALIDACIÓN:
  - acción: eval/frases_validacion.csv y eval/frases_objetos_validacion.csv (escritas a mano);
  - objeto: ml/intencion/frases_validacion_generada.csv (objetos reservados, que no están en el entrenamiento)
    y eval/frases_objetos_validacion.csv.
La PRUEBA (eval/frases_prueba.csv, frases_nuevas.csv y frases_objetos_prueba.csv) se mide de punta a punta con
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
    # "balanceada" pesa cada clase por igual; sin balancear, pesa más "mostrar", que tiene más frases (y es el
    # pedido más común)
    "logistica_balanceada": lambda: LogisticRegression(C=10, max_iter=3000, class_weight="balanced"),
    "logistica": lambda: LogisticRegression(C=10, max_iter=3000),
    "mlp_256": lambda: MLPClassifier((256,), alpha=1e-3, max_iter=600, early_stopping=True, random_state=0),
    "mlp_512_256": lambda: MLPClassifier((512, 256), alpha=1e-3, max_iter=600, early_stopping=True, random_state=0),
}
TAMANO = {"logistica": 0, "logistica_balanceada": 0, "mlp_256": 1, "mlp_512_256": 2}
ETIQUETADORES = {
    "contexto_logistica": lambda: LogisticRegression(C=1, max_iter=2000),
    "contexto_mlp_256": lambda: MLPClassifier((256,), alpha=1e-4, max_iter=200, early_stopping=True,
                                              random_state=0),
    "contexto_mlp_512": lambda: MLPClassifier((512,), alpha=1e-4, max_iter=200, early_stopping=True,
                                              random_state=0),
}
TAMANO_ETIQUETADOR = {"rasgos": 0, "contexto_logistica": 1, "combinado_logistica": 1.5, "contexto_mlp_256": 2,
                      "combinado_mlp_256": 2.5, "contexto_mlp_512": 3, "combinado_mlp_512": 3.5}
TOLERANCIA = 0.002
# Los rasgos con peso menor no cambian ningún objeto de validación y ocupan el 80 % del archivo (70 mil -> 13 mil)
PESO_MINIMO = 0.05


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def codificar(modelo, frases, nombre):
    prefijo = intent_modelo.prefijo_consulta(nombre)
    return modelo.encode([prefijo + f for f in frases], normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False)


def capas(clf):
    """Pesos de cualquier clasificador como capas densas (la logística es una sola capa)."""
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


def con_objeto(filas):
    return [f for f in filas if f["accion"] in intent_modelo.ACCIONES_CON_OBJETO]


def entrenar_rasgos(filas):
    from sklearn.feature_extraction import DictVectorizer
    X, y = [], []
    for fila in con_objeto(filas):
        palabras, marcas = etiquetas_objeto(fila["frase"], fila["objeto"])
        X += [intent_modelo.rasgos(palabras, i, fila["accion"]) for i in range(len(palabras))]
        y += marcas
    vec = DictVectorizer()
    clf = LogisticRegression(C=3, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(X), y)
    pesos = {nombre: round(float(w), 5) for nombre, w in zip(vec.get_feature_names_out(), clf.coef_[0])
             if abs(w) > PESO_MINIMO}
    return {"tipo": "rasgos", "pesos": pesos, "sesgo": round(float(clf.intercept_[0]), 5)}


def matriz_contexto(filas, vectores):
    X, y = [], []
    for fila in filas:
        _, marcas = etiquetas_objeto(fila["frase"], fila["objeto"])
        v = vectores[fila["frase"]]
        assert len(v) == len(marcas), fila["frase"]
        X.append(intent_modelo.rasgos_contexto(v, fila["accion"]))
        y += marcas
    return np.vstack(X), np.array(y)


def exactitud_objeto(interprete, filas, vectores=None):
    """Proporción de frases en las que el objeto sale exacto (con la acción correcta dada)."""
    aciertos = 0
    for f in filas:
        v = vectores[f["frase"]] if vectores is not None else None
        o = interprete.objeto(f["frase"], f["accion"], v) or ""
        aciertos += intent_modelo.palabras(o) == intent_modelo.palabras(f["objeto"])
    return aciertos / len(filas)


def main():
    from sentence_transformers import SentenceTransformer

    entrenamiento = leer(AQUI / "frases_entrenamiento.csv")
    validacion_generada = leer(AQUI / "frases_validacion_generada.csv")
    validacion = leer(RAIZ / "eval" / "frases_validacion.csv") + \
        [{"frase": f["frase"], "accion": f["accion"]} for f in leer(RAIZ / "eval" / "frases_objetos_validacion.csv")]
    objetos_validacion = leer(RAIZ / "eval" / "frases_objetos_validacion.csv")
    prueba = leer(RAIZ / "eval" / "frases_prueba.csv")
    acciones = sorted({f["accion"] for f in entrenamiento})
    y_ent = [f["accion"] for f in entrenamiento]
    y_val = [f["accion"] for f in validacion]
    y_pru = [f["accion"] for f in prueba]
    print(f"entrenamiento {len(entrenamiento)} · validación {len(validacion)} · prueba {len(prueba)} · "
          f"{len(acciones)} acciones · validación de objetos {len(validacion_generada)} generadas + "
          f"{len(objetos_validacion)} a mano")

    # 1) Acción: embeddings x cabezales, elegido con validación (a igual exactitud, el más chico)
    resultados, mejor = [], None
    for nombre in EMBEDDINGS:
        modelo = SentenceTransformer(nombre)
        t0 = time.perf_counter()
        E_ent = codificar(modelo, [f["frase"] for f in entrenamiento], nombre)
        print(f"  {nombre}: {len(entrenamiento)} frases codificadas en {time.perf_counter() - t0:.0f} s")
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
            print(f"  {nombre.split('/')[-1]:38s} {cabezal:12s} validación {fila['exactitud_validacion']:.3f} "
                  f"(F1 {fila['f1_validacion']:.3f}) · prueba {fila['exactitud_prueba']:.3f} · {fila['segundos']} s")
            clave = (fila["exactitud_validacion"], fila["f1_validacion"], -TAMANO[cabezal])
            if mejor is None or clave > mejor[0]:
                mejor = (clave, nombre, cabezal, clf, p_val)

    _, nombre, cabezal, clf, p_val = mejor
    print(f"\nelegido para la acción: {nombre} + {cabezal}")
    for f, p in zip(validacion, p_val):
        if p != f["accion"]:
            print(f"  validación, falla: {f['frase']!r} → {p} (era {f['accion']})")

    # 2) Etiquetador del objeto, con los vectores de palabra del modelo elegido
    modelo = SentenceTransformer(nombre)
    prefijo = intent_modelo.prefijo_consulta(nombre)
    ent_obj, val_gen = con_objeto(entrenamiento), validacion_generada
    t0 = time.perf_counter()
    vectores = {}
    for filas in (ent_obj, val_gen, objetos_validacion):
        frases = [f["frase"] for f in filas]
        for frase, (_, v) in zip(frases, intent_modelo.codificar(modelo, prefijo, frases)):
            vectores[frase] = v
    print(f"vectores de palabra de {len(vectores)} frases en {time.perf_counter() - t0:.0f} s")

    base = {"embeddings": nombre, "acciones": [str(c) for c in clf.classes_],
            "capas": [{"W": intent_modelo.empaquetar(c["W"]), "b": intent_modelo.empaquetar(c["b"])}
                      for c in capas(clf)]}

    def probar(etiquetador):
        interprete = intent_modelo.InterpreteEntrenado({**base, "etiquetador": etiquetador}, modelo)
        usa = vectores if etiquetador["tipo"] != "rasgos" else None
        return (round(exactitud_objeto(interprete, val_gen, usa), 4),
                round(exactitud_objeto(interprete, objetos_validacion, usa), 4))

    comparacion, elegido = [], None
    t0 = time.perf_counter()
    de_rasgos = entrenar_rasgos(entrenamiento)
    candidatos = [("rasgos", de_rasgos)]
    X, y = matriz_contexto(ent_obj, vectores)
    print(f"etiquetador: {X.shape[0]} palabras × {X.shape[1]} rasgos ({y.mean():.0%} son parte del objeto)")
    tiempos = {"rasgos": time.perf_counter() - t0}
    for nombre_et, crear in ETIQUETADORES.items():
        t0 = time.perf_counter()
        et = crear().fit(X, y)
        tiempos[nombre_et] = time.perf_counter() - t0
        pesos = [{"W": intent_modelo.empaquetar(c["W"]), "b": intent_modelo.empaquetar(c["b"])} for c in capas(et)]
        candidatos.append((nombre_et, {"tipo": "contexto", "capas": pesos}))
        combinado = nombre_et.replace("contexto", "combinado")
        tiempos[combinado] = tiempos[nombre_et] + tiempos["rasgos"]
        candidatos.append((combinado, {"tipo": "combinado", "capas": pesos,
                                       "rasgos": {"pesos": de_rasgos["pesos"], "sesgo": de_rasgos["sesgo"]}}))
    for nombre_et, etiquetador in candidatos:
        generada, a_mano = probar(etiquetador)
        fila = {"etiquetador": nombre_et, "objeto_exacto_generadas": generada, "objeto_exacto_a_mano": a_mano,
                "segundos": round(tiempos[nombre_et], 1)}
        comparacion.append(fila)
        print(f"  {nombre_et:20s} objeto exacto: generadas {generada:.3f} · a mano {a_mano:.3f} · "
              f"{fila['segundos']} s")
    promedio = {f["etiquetador"]: (f["objeto_exacto_generadas"] + f["objeto_exacto_a_mano"]) / 2 for f in comparacion}
    mejor_promedio = max(promedio.values())
    nombre_et = min((n for n in promedio if promedio[n] >= mejor_promedio - TOLERANCIA),
                    key=TAMANO_ETIQUETADOR.get)
    etiquetador = dict(candidatos)[nombre_et]
    print(f"elegido para el objeto: {nombre_et}")
    interprete = intent_modelo.InterpreteEntrenado({**base, "etiquetador": etiquetador}, modelo)
    for f in objetos_validacion:
        o = interprete.objeto(f["frase"], f["accion"], vectores[f["frase"]] if etiquetador["tipo"] != "rasgos"
                              else None)
        if intent_modelo.palabras(o or "") != intent_modelo.palabras(f["objeto"]):
            print(f"  validación, objeto: {f['frase']!r} → {o!r} (era {f['objeto']!r})")

    # 3) Exportar: pesos en JSON (el backend hace la cuenta con numpy, sin pickle ni scikit-learn)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    datos = {
        **base,
        "activacion_oculta": "relu",
        "etiquetador": {**etiquetador, "nombre": nombre_et},
        "entrenamiento": {"frases": len(entrenamiento), "fecha": date.today().isoformat(),
                          "origen": "ml/intencion/generar_frases.py"},
    }
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"modelo -> {SALIDA.relative_to(RAIZ)} ({SALIDA.stat().st_size // 1024} KB)")

    REPORTE.parent.mkdir(exist_ok=True)
    REPORTE.write_text(json.dumps({
        "fecha": date.today().isoformat(),
        "frases": {"entrenamiento": len(entrenamiento), "validacion_accion": len(validacion),
                   "validacion_objeto_generadas": len(val_gen), "validacion_objeto_a_mano": len(objetos_validacion),
                   "prueba": len(prueba)},
        "comparacion_accion": resultados,
        "comparacion_objeto": comparacion,
        "elegido": {"embeddings": nombre, "cabezal": cabezal, "etiquetador": nombre_et},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
