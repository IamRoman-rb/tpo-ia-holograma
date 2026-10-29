"""Árboles de decisión para los gestos: aprenden reglas legibles con HaGRID y se comparan con las escritas a mano.

Las reglas de frontend/js/gestures.js son umbrales sobre unas pocas métricas de la mano, calibrados a ojo. Un
árbol de decisión aprende esos mismos umbrales (y cuáles preguntar primero) de 196 mil manos, eligiendo en
cada nodo la pregunta que más reduce la impureza de Gini. Se comparan:
  - árbol de decisión: profundidad y poda por costo-complejidad (ccp_alpha) elegidas con validación;
    es el que se exporta al visor (clasificador "arbol"), porque cada decisión se puede leer entera;
  - bosque aleatorio (random forest): 300 árboles con muestras y rasgos al azar, votando;
  - gradient boosting: árboles chicos en serie, cada uno corrige los errores de los anteriores;
  - el mismo bosque sobre las 63 coordenadas crudas (¿cuánto aportan los rasgos legibles?).
Rasgos: ml/arboles/rasgos_arbol.py (19: qué tan estirado está cada dedo, ángulos de las articulaciones y
distancias entre puntas), iguales a frontend/js/arbol.js. Misma partición por persona que la red
(preparar.particion), elección con VALIDACIÓN, resultado en PRUEBA (personas no vistas), con intervalos por
bootstrap y McNemar contra las reglas y contra la red v2.

Uso (desde la raíz):  backend/.venv/Scripts/python ml/arboles/entrenar_arboles.py [--datos ml/data/hagrid]
Salida: frontend/ml/arbol_gestos.json, frontend/tests/datos/arbol_casos.json, ml/resultados/arboles_reporte.json
y ml/resultados/arbol_reglas.txt (las reglas aprendidas por un árbol de profundidad 4, para leerlas).
"""
import argparse
import base64
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
ML = AQUI.parent
RAIZ = ML.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ / "eval"))
import preparar  # noqa: E402
from entrenar_red_gestos import CLASES, elegir_umbral, reglas_en, resumen, sesgos  # noqa: E402
from estadistica import intervalo_bootstrap, mcnemar  # noqa: E402
from rasgos_arbol import RASGOS, rasgos  # noqa: E402

RESULTADOS = ML / "resultados"
MODELO_WEB = RAIZ / "frontend" / "ml" / "arbol_gestos.json"
RED_WEB = RAIZ / "frontend" / "ml" / "mlp_gestos.json"
CASOS_TEST = RAIZ / "frontend" / "tests" / "datos" / "arbol_casos.json"
PROFUNDIDADES = [3, 4, 6, 8, 10, 12, 15, 18, 22, None]
ALFAS_PODA = [0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4]


def f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def con_umbral(proba, umbral):
    k = proba.argmax(1)
    return np.where(proba.max(1) >= umbral, k, CLASES.index("ninguno"))


def red_v2(X):
    """Probabilidades de la red del visor (frontend/ml/mlp_gestos.json), para compararla sobre las mismas manos."""
    modelo = json.loads(RED_WEB.read_text(encoding="utf-8"))

    def matriz(m):
        if isinstance(m, list):
            return np.asarray(m, np.float32)
        return np.frombuffer(base64.b64decode(m["float32"]), "<f4").reshape(m["forma"])
    a = X
    for k, c in enumerate(modelo["capas"]):
        a = a @ matriz(c["W"]) + matriz(c["b"])
        if k < len(modelo["capas"]) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return np.where(p.max(1) >= modelo["umbral_confianza"], p.argmax(1), CLASES.index("ninguno"))


def exportar_arbol(arbol, umbral):
    t = arbol.tree_
    valor = t.value[:, 0, :]
    valor = valor / valor.sum(axis=1, keepdims=True)
    return {
        "clases": [CLASES[int(c)] for c in arbol.classes_], "rasgos": RASGOS, "umbral_confianza": umbral,
        "profundidad": int(arbol.get_depth()), "hojas": int(arbol.get_n_leaves()),
        "nodos": {"izq": t.children_left.tolist(), "der": t.children_right.tolist(),
                  "rasgo": [int(max(r, 0)) for r in t.feature],
                  "umbral": [round(float(u), 6) for u in t.threshold],
                  "valor": np.round(valor, 4).tolist()},
        "origen": "HaGRID (CC BY-SA 4.0) · ml/arboles/entrenar_arboles.py", "fecha": date.today().isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=preparar.DATOS_HAGRID)
    args = ap.parse_args()

    df = preparar.cargar(args.datos)
    parte = df["persona"].map(preparar.particion).to_numpy()
    X63 = preparar.normalizar(df)
    X = rasgos(X63)
    y = np.array([CLASES.index(c) for c in df["clase"]])
    ent, val, pru = parte == "entrenamiento", parte == "validacion", parte == "prueba"
    print(f"manos: entrenamiento {ent.sum()} · validación {val.sum()} · prueba {pru.sum()} · {X.shape[1]} rasgos")

    # 1) Árbol de decisión: profundidad (a igual F1 se prefiere el más profundo: más preciso, no importa el tamaño)
    tabla_profundidad, prof_elegida, f1_elegida = [], None, -1
    for prof in PROFUNDIDADES:
        t0 = time.perf_counter()
        arbol = DecisionTreeClassifier(max_depth=prof, class_weight="balanced", min_samples_leaf=5, random_state=0)
        arbol.fit(X[ent], y[ent])
        f = f1(y[val], arbol.predict(X[val]))
        fila = {"profundidad": prof or "sin límite", "hojas": int(arbol.get_n_leaves()),
                "profundidad_real": int(arbol.get_depth()), "f1_validacion": round(f, 4),
                "segundos": round(time.perf_counter() - t0, 1)}
        tabla_profundidad.append(fila)
        print(f"  árbol profundidad {str(prof):>4s}: {fila['hojas']:>6} hojas · F1 validación {f:.4f}")
        if f >= f1_elegida - 1e-4:   # las profundidades van de menor a mayor: el empate lo gana la más profunda
            prof_elegida, f1_elegida = prof, max(f, f1_elegida)

    # 2) Poda por costo-complejidad: saca las ramas que casi no mejoran la impureza (menos sobreajuste)
    tabla_poda, elegido = [], None
    for alfa in ALFAS_PODA:
        arbol = DecisionTreeClassifier(max_depth=prof_elegida, class_weight="balanced", min_samples_leaf=5,
                                       ccp_alpha=alfa, random_state=0).fit(X[ent], y[ent])
        f = f1(y[val], arbol.predict(X[val]))
        tabla_poda.append({"ccp_alpha": alfa, "hojas": int(arbol.get_n_leaves()), "f1_validacion": round(f, 4)})
        print(f"  poda ccp_alpha {alfa:<7g}: {arbol.get_n_leaves():>6} hojas · F1 validación {f:.4f}")
        if elegido is None or f > elegido[1] + 1e-4:
            elegido = (alfa, f, arbol)
    alfa, f1_arbol, arbol = elegido
    umbral_arbol, _ = elegir_umbral(arbol.predict_proba(X[val]), y[val])
    print(f"árbol elegido: profundidad {arbol.get_depth()}, {arbol.get_n_leaves()} hojas, ccp_alpha {alfa}, "
          f"umbral de confianza {umbral_arbol}")

    # 3) Ensambles de árboles
    modelos = {}
    t0 = time.perf_counter()
    bosque = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=0).fit(X[ent], y[ent])
    modelos["bosque aleatorio (300 árboles)"] = (bosque, X, time.perf_counter() - t0)
    t0 = time.perf_counter()
    boosting = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                              class_weight="balanced", early_stopping=True,
                                              validation_fraction=0.1, random_state=0).fit(X[ent], y[ent])
    modelos[f"gradient boosting ({boosting.n_iter_} rondas)"] = (boosting, X, time.perf_counter() - t0)
    t0 = time.perf_counter()
    bosque63 = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0).fit(X63[ent], y[ent])
    modelos["bosque sobre las 63 coordenadas"] = (bosque63, X63, time.perf_counter() - t0)

    # 4) Prueba: todos sobre las mismas manos, cada uno con su umbral elegido en validación
    y_pru = np.array(CLASES)[y[pru]]
    predicciones = {"reglas": reglas_en(df[pru].reset_index(drop=True))}
    predicciones[f"árbol de decisión (prof. {arbol.get_depth()})"] = \
        np.array(CLASES)[con_umbral(arbol.predict_proba(X[pru]), umbral_arbol)]
    umbrales = {}
    for nombre, (m, entrada, segundos) in modelos.items():
        u, _ = elegir_umbral(m.predict_proba(entrada[val]), y[val])
        umbrales[nombre] = {"umbral": u, "segundos_entrenamiento": round(segundos)}
        predicciones[nombre] = np.array(CLASES)[con_umbral(m.predict_proba(entrada[pru]), u)]
    if RED_WEB.exists():
        predicciones["red neuronal v2 (visor)"] = np.array(CLASES)[red_v2(X63[pru])]

    comparacion = []
    for nombre, pred in predicciones.items():
        fila = resumen(nombre, y_pru, pred)
        media, desde, hasta = intervalo_bootstrap(pred == y_pru, repeticiones=2000)
        fila["exactitud_ic95"] = [round(desde, 4), round(hasta, 4)]
        comparacion.append(fila)
    import pandas as pd
    print("\n" + pd.DataFrame(comparacion).to_string(index=False))

    nombre_arbol = next(n for n in predicciones if n.startswith("árbol"))
    pruebas_mcnemar = []
    for otro in predicciones:
        if otro != nombre_arbol:
            solo_a, solo_b, p = mcnemar(predicciones[nombre_arbol] == y_pru, predicciones[otro] == y_pru)
            pruebas_mcnemar.append({"a": nombre_arbol, "b": otro, "solo_a": solo_a, "solo_b": solo_b, "p": p})
            print(f"  McNemar {nombre_arbol} vs {otro}: {solo_a} / {solo_b} · p = {p:.2g}")

    # 5) Reglas legibles: un árbol de profundidad 4 (lo que se puede contar en una diapositiva)
    chico = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=50, random_state=0)
    chico.fit(X[ent], y[ent])
    f1_chico = f1(y[pru], chico.predict(X[pru]))
    reglas_txt = export_text(chico, feature_names=RASGOS, class_names=[CLASES[int(c)] for c in chico.classes_],
                             decimals=2)
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "arbol_reglas.txt").write_text(
        f"Árbol de profundidad 4 entrenado con HaGRID (F1 macro en prueba {f1_chico:.3f}).\n\n" + reglas_txt,
        encoding="utf-8")

    # 6) Importancia de cada rasgo (cuánto reduce la impureza en el bosque) y exportación
    importancia = sorted(zip(RASGOS, bosque.feature_importances_), key=lambda r: -r[1])
    print("rasgos más importantes (bosque): " + ", ".join(f"{r} {v:.3f}" for r, v in importancia[:6]))
    web = exportar_arbol(arbol, umbral_arbol)
    MODELO_WEB.write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rng = np.random.default_rng(0)
    muestras = rng.choice(np.flatnonzero(pru), 12, replace=False)
    casos = [{"x": np.round(X63[i], 6).tolist(), "rasgos": np.round(X[i], 5).tolist(),
              "proba": np.round(arbol.predict_proba(X[i:i + 1])[0], 4).tolist()} for i in muestras]
    CASOS_TEST.write_text(json.dumps({"clases": web["clases"], "casos": casos}), encoding="utf-8")

    reporte = {
        "fecha": date.today().isoformat(), "dataset": "HaGRID, CC BY-SA 4.0",
        "filas": {"entrenamiento": int(ent.sum()), "validacion": int(val.sum()), "prueba": int(pru.sum())},
        "rasgos": RASGOS, "profundidad_validacion": tabla_profundidad, "poda_validacion": tabla_poda,
        "arbol_elegido": {"profundidad": int(arbol.get_depth()), "hojas": int(arbol.get_n_leaves()),
                          "ccp_alpha": alfa, "f1_validacion": round(f1_arbol, 4), "umbral_confianza": umbral_arbol},
        "ensambles": umbrales, "comparacion_prueba": comparacion, "mcnemar_prueba": pruebas_mcnemar,
        "arbol_profundidad_4": {"f1_prueba": round(f1_chico, 4), "reglas": "ml/resultados/arbol_reglas.txt"},
        "importancia_rasgos_bosque": {r: round(float(v), 4) for r, v in importancia},
        "sesgos_prueba_arbol": sesgos(df[pru].reset_index(drop=True), y_pru, predicciones[nombre_arbol]),
        "por_clase_arbol": classification_report(y_pru, predicciones[nombre_arbol], output_dict=True,
                                                 zero_division=0),
    }
    (RESULTADOS / "arboles_reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"\n{MODELO_WEB.relative_to(RAIZ)} ({MODELO_WEB.stat().st_size // 1024} KB) · "
          f"reporte en ml/resultados/arboles_reporte.json")


if __name__ == "__main__":
    main()
