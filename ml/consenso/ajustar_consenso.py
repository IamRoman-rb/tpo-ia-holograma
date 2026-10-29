"""Ajusta los pesos del consenso de motores (backend/consenso.py) en validación.

Cada motor propio interpreta las frases de validación una sola vez y se guardan sus distribuciones sobre las
acciones. Después se comparan formas de combinarlas, todas sobre las mismas frases:
  - el mejor motor solo;
  - voto simple: gana la acción que más motores eligieron (empate: la de mayor confianza sumada);
  - promedio de probabilidades;
  - log-lineal con pesos por máxima verosimilitud: se minimiza −Σ log P(acción correcta) + λ·||w||² con
    L-BFGS-B (scipy), pesos ≥ 0 y un sesgo por acción. Es una regresión logística sobre los log-p de los motores;
  - log-lineal con pesos hallados por un algoritmo genético (la misma idea que ml/evolucion/neuroevolucion.py,
    acá sobre 5 números): sirve para comparar un optimizador por gradiente con uno sin gradiente.
Como los pesos se ajustan con las mismas frases con las que se mide, la exactitud en validación se calcula con
validación cruzada de 5 particiones (se ajusta con 4/5 y se mide en el 1/5 restante). La prueba de punta a punta
es eval/eval_intent.py --motor consenso.

Uso (desde la raíz):  backend/.venv/Scripts/python ml/consenso/ajustar_consenso.py
Salida: backend/modelos/consenso.json y ml/resultados/consenso_reporte.json
"""
import csv
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))
import consenso  # noqa: E402
from estadistica import ece, intervalo_bootstrap  # noqa: E402

REPORTE = RAIZ / "ml" / "resultados" / "consenso_reporte.json"
CONJUNTOS = ["frases_validacion.csv", "frases_objetos_validacion.csv", "frases_comandos_validacion.csv"]
LAMBDA = 0.01


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [(fila["frase"], fila["accion"]) for fila in csv.DictReader(f)]


def nll(x, L, y, motores, n_acc, lam=LAMBDA):
    """L: (frases, motores, acciones) log-probabilidades. x = pesos (motores) + sesgos (acciones)."""
    w, b = x[:len(motores)], x[len(motores):]
    z = np.einsum("fma,m->fa", L, w) + b
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -log_p[np.arange(len(y)), y].mean() + lam * (w ** 2).sum() + lam * (b ** 2).sum()


def ajustar_lbfgs(L, y, motores, n_acc):
    x0 = np.concatenate([np.full(len(motores), 1.0 / len(motores)), np.zeros(n_acc)])
    limites = [(0, None)] * len(motores) + [(None, None)] * n_acc
    r = minimize(nll, x0, args=(L, y, motores, n_acc), method="L-BFGS-B", bounds=limites)
    return r.x


def ajustar_genetico(L, y, motores, n_acc, generaciones=60, poblacion=40, semilla=0):
    """Algoritmo genético sobre los pesos (sin sesgos): torneo, cruce aritmético, mutación gaussiana, elitismo."""
    r = np.random.default_rng(semilla)
    aptitud = lambda w: -nll(np.concatenate([w, np.zeros(n_acc)]), L, y, motores, n_acc)  # noqa: E731
    pob = r.uniform(0, 1.5, size=(poblacion, len(motores)))
    for _ in range(generaciones):
        apt = np.array([aptitud(w) for w in pob])
        orden = np.argsort(apt)[::-1]
        nueva = [pob[orden[0]], pob[orden[1]]]
        while len(nueva) < poblacion:
            a, b = (pob[max(r.choice(poblacion, 3), key=lambda i: apt[i])] for _ in range(2))
            t = r.uniform(size=len(motores))
            hijo = t * a + (1 - t) * b + r.normal(0, 0.1, size=len(motores)) * (r.uniform(size=len(motores)) < 0.3)
            nueva.append(np.clip(hijo, 0, 5))
        pob = np.array(nueva)
    apt = np.array([aptitud(w) for w in pob])
    return np.concatenate([pob[int(np.argmax(apt))], np.zeros(n_acc)])


def predecir(L, x, motores):
    w, b = x[:len(motores)], x[len(motores):]
    return (np.einsum("fma,m->fa", L, w) + b).argmax(axis=1)


def probabilidades(L, x, motores):
    w, b = x[:len(motores)], x[len(motores):]
    z = np.einsum("fma,m->fa", L, w) + b
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def validacion_cruzada(L, y, motores, n_acc, ajustar, k=5):
    orden = np.random.default_rng(0).permutation(len(y))
    pred = np.zeros(len(y), int)
    proba = np.zeros((len(y), n_acc))
    for parte in np.array_split(orden, k):
        resto = np.setdiff1d(orden, parte)
        x = ajustar(L[resto], y[resto], motores, n_acc)
        pred[parte] = predecir(L[parte], x, motores)
        proba[parte] = probabilidades(L[parte], x, motores)
    return pred, proba


def main():
    acciones = consenso.acciones()
    frases = [x for nombre in CONJUNTOS for x in leer(RAIZ / "eval" / nombre)]
    t0 = time.perf_counter()
    salidas = []
    for i, (frase, _) in enumerate(frases):
        salidas.append(consenso.correr_motores(frase))
        if i % 25 == 0:
            print(f"  {i + 1}/{len(frases)} frases · {time.perf_counter() - t0:.0f} s", flush=True)
    motores = [m for m in consenso.MOTORES if all(m in s for s in salidas)]
    print(f"motores que respondieron siempre: {motores}")
    L = np.log(np.array([[s[m]["distribucion"] for m in motores] for s in salidas]))
    y = np.array([acciones.index(a) for _, a in frases])
    n_acc = len(acciones)

    tabla = []

    def fila(nombre, pred, proba=None):
        acierto = pred == y
        media, desde, hasta = intervalo_bootstrap(acierto, repeticiones=4000)
        f = {"metodo": nombre, "exactitud": round(media, 4), "ic95": [round(desde, 4), round(hasta, 4)]}
        if proba is not None:
            f["ece"] = round(ece(proba.max(axis=1), acierto), 4)
        tabla.append(f)
        print(f"  {nombre:45s} {100 * media:5.1f} % [{100 * desde:.0f}–{100 * hasta:.0f}]"
              + (f" · ECE {f['ece']:.3f}" if proba is not None else ""))

    for k, m in enumerate(motores):
        fila(f"solo {m}", L[:, k].argmax(axis=1), np.exp(L[:, k]))
    votos = np.zeros((len(y), n_acc))
    for k in range(len(motores)):
        votos[np.arange(len(y)), L[:, k].argmax(axis=1)] += 1 + 1e-3 * np.exp(L[:, k]).max(axis=1)
    fila("voto simple", votos.argmax(axis=1))
    promedio = np.exp(L).mean(axis=1)
    fila("promedio de probabilidades", promedio.argmax(axis=1), promedio)
    pred, proba = validacion_cruzada(L, y, motores, n_acc, ajustar_lbfgs)
    fila("log-lineal, L-BFGS (validación cruzada)", pred, proba)
    pred, proba = validacion_cruzada(L, y, motores, n_acc, ajustar_genetico)
    fila("log-lineal, algoritmo genético (validación cruzada)", pred, proba)

    # Pesos finales con todas las frases de validación (los que usa el backend)
    x = ajustar_lbfgs(L, y, motores, n_acc)
    x_ga = ajustar_genetico(L, y, motores, n_acc)
    pesos = {m: round(float(w), 4) for m, w in zip(motores, x[:len(motores)])}
    sesgos = {a: round(float(b), 4) for a, b in zip(acciones, x[len(motores):])}
    print("pesos: " + ", ".join(f"{m} {w:.2f}" for m, w in pesos.items()))
    (consenso.RUTA_PESOS).write_text(json.dumps({
        "metodo": "log-lineal, máxima verosimilitud (L-BFGS-B)", "pesos": pesos, "sesgos": sesgos,
        "lambda": LAMBDA, "frases": len(y), "fecha": date.today().isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    REPORTE.write_text(json.dumps({
        "fecha": date.today().isoformat(), "frases_validacion": len(y), "motores": motores, "comparacion": tabla,
        "pesos_lbfgs": pesos, "sesgos_lbfgs": sesgos,
        "pesos_genetico": {m: round(float(w), 4) for m, w in zip(motores, x_ga[:len(motores)])},
        "ms_por_frase": round((time.perf_counter() - t0) * 1000 / len(y)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {consenso.RUTA_PESOS.relative_to(RAIZ)} · {REPORTE.relative_to(RAIZ)}")


if __name__ == "__main__":
    random.seed(0)
    main()
