"""Neuroevolución: un algoritmo genético diseña la red de gestos (arquitectura e hiperparámetros).

La red v2 (ml/entrenar_red_gestos.py) se eligió entre 5 arquitecturas escritas a mano. Acá se busca en un
espacio mucho más grande con un algoritmo genético propio:
  - genoma: profundidad (1 a 8 capas ocultas), ancho, conexiones residuales, normalización por capa (LayerNorm),
    activación (ReLU, GELU, SiLU), dropout, tasa de aprendizaje, weight decay, tamaño del lote e intensidad del
    aumento de datos;
  - aptitud: F1 macro en VALIDACIÓN después de un entrenamiento corto (un subconjunto de las manos y pocas
    épocas, siempre igual para todos), así se pueden evaluar cientos de redes;
  - selección por torneo (gana la mejor de 3 al azar), cruce uniforme (cada gen sale de uno de los dos padres),
    mutación (cada gen cambia con probabilidad 0,25: un valor vecino o uno nuevo al azar) y elitismo (las 2
    mejores pasan intactas). Los genomas ya evaluados no se vuelven a entrenar (caché de aptitudes).
Comparación justa: una búsqueda al azar con el MISMO presupuesto de redes entrenadas. Si el genético no le gana
al azar, no aporta nada.
Al final, los 3 mejores genomas se entrenan completos (todas las manos, hasta 60 épocas); se elige con
validación y se informa en PRUEBA contra la red v2, con McNemar. Si gana en validación, se exporta al visor
(frontend/ml/mlp_gestos.json, versión 3: frontend/js/mlp.js lee capas residuales, LayerNorm y GELU/SiLU).

Uso (desde la raíz, con el entorno del backend, que tiene PyTorch):
    backend/.venv/Scripts/python ml/evolucion/neuroevolucion.py [--generaciones 10] [--poblacion 14] [--rapido]
Salida: ml/resultados/neuroevolucion_reporte.json, ml/resultados/neuroevolucion_curva.csv y, si gana,
frontend/ml/mlp_gestos.json + frontend/tests/datos/mlp_casos.json
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
ML = AQUI.parent
RAIZ = ML.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(RAIZ / "eval"))
import preparar  # noqa: E402
from entrenar_red_gestos import (CLASES, MODELO_WEB, CASOS_TEST, con_umbral, elegir_umbral, empaquetar,  # noqa: E402
                                 resumen)
from estadistica import intervalo_bootstrap, mcnemar  # noqa: E402

RESULTADOS = ML / "resultados"

# ---------------------------------------------------------------- genoma

GENES = {
    "profundidad": [1, 2, 3, 4, 5, 6, 8],
    "ancho": [64, 128, 192, 256, 384, 512, 768, 1024],
    "residual": [False, True],
    "normalizacion": [False, True],
    "activacion": ["relu", "gelu", "silu"],
    "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
    "tasa": [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    "lote": [256, 512, 1024, 2048],
    "aumento": [0.0, 0.5, 1.0, 1.5],
}


def genoma_al_azar(r):
    return {g: r.choice(v) for g, v in GENES.items()}


def clave(genoma):
    return json.dumps(genoma, sort_keys=True)


def cruzar(a, b, r):
    """Cruce uniforme: cada gen viene de uno de los padres, al azar."""
    return {g: (a if r.random() < 0.5 else b)[g] for g in GENES}


def mutar(genoma, r, prob=0.25):
    """Cada gen cambia con probabilidad prob: la mitad de las veces a un valor vecino (un paso chico en una
    escala ordenada) y la otra mitad a cualquier valor."""
    hijo = dict(genoma)
    for g, valores in GENES.items():
        if r.random() < prob:
            i = valores.index(hijo[g])
            if r.random() < 0.5 and len(valores) > 2:
                i = min(len(valores) - 1, max(0, i + r.choice([-1, 1])))
            else:
                i = r.randrange(len(valores))
            hijo[g] = valores[i]
    return hijo


def torneo(poblacion, aptitud, r, k=3):
    return max(r.sample(poblacion, k), key=lambda g: aptitud[clave(g)])


# ---------------------------------------------------------------- red

ACTIVACIONES = {"relu": torch.nn.ReLU, "gelu": lambda: torch.nn.GELU(approximate="tanh"), "silu": torch.nn.SiLU}


class Bloque(torch.nn.Module):
    def __init__(self, ancho, g):
        super().__init__()
        self.norma = torch.nn.LayerNorm(ancho) if g["normalizacion"] else None
        self.lineal = torch.nn.Linear(ancho, ancho)
        self.act = ACTIVACIONES[g["activacion"]]()
        self.drop = torch.nn.Dropout(g["dropout"])
        self.residual = g["residual"]

    def forward(self, x):
        h = self.drop(self.act(self.lineal(self.norma(x) if self.norma is not None else x)))
        return x + h if self.residual else h


class Red(torch.nn.Module):
    """entrada (63 → ancho) → (profundidad − 1) bloques de ancho fijo, con o sin residual y LayerNorm → salida."""

    def __init__(self, g):
        super().__init__()
        self.genoma = g
        self.entrada = torch.nn.Linear(63, g["ancho"])
        self.act = ACTIVACIONES[g["activacion"]]()
        self.drop = torch.nn.Dropout(g["dropout"])
        self.bloques = torch.nn.ModuleList(Bloque(g["ancho"], g) for _ in range(g["profundidad"] - 1))
        self.norma_salida = torch.nn.LayerNorm(g["ancho"]) if g["normalizacion"] else None
        self.salida = torch.nn.Linear(g["ancho"], len(CLASES))

    def forward(self, x):
        x = self.drop(self.act(self.entrada(x)))
        for b in self.bloques:
            x = b(x)
        return self.salida(self.norma_salida(x) if self.norma_salida is not None else x)


def parametros(g):
    a = g["ancho"]
    n = 63 * a + a + (g["profundidad"] - 1) * (a * a + a) + a * len(CLASES) + len(CLASES)
    return n + (2 * a * g["profundidad"] if g["normalizacion"] else 0)


def aumentar(x, gen, intensidad):
    """El aumento de la red v2 (espejo, rotación, giro en profundidad, ruido), con la intensidad del genoma."""
    if intensidad == 0:
        return x
    n = x.shape[0]
    p = x.view(n, 21, 3).clone()
    espejo = torch.rand(n, generator=gen) < 0.5
    p[espejo, :, 0] *= -1
    a = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(25 * intensidad)
    b = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(15 * intensidad)
    ca, sa, cb, sb = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b)
    x0, y0, z0 = p[..., 0], p[..., 1], p[..., 2]
    x1 = ca[:, None] * x0 - sa[:, None] * y0
    y1 = sa[:, None] * x0 + ca[:, None] * y0
    x2 = cb[:, None] * x1 + sb[:, None] * z0
    z2 = -sb[:, None] * x1 + cb[:, None] * z0
    p = torch.stack([x2, y1, z2], dim=-1)
    p += torch.randn(p.shape, generator=gen) * 0.02 * intensidad
    p -= p[:, :1, :].clone()
    escala = p[:, 9, :].norm(dim=1).clamp_min(1e-6)
    return (p / escala[:, None, None]).reshape(n, 63)


def probabilidades(red, X, lote=8192):
    red.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(red(X[i:i + lote]), 1) for i in range(0, len(X), lote)]).numpy()


def entrenar(g, X, y, Xv, yv, epocas, paciencia=None, semilla=0):
    """Entrena la red del genoma. Devuelve (red, F1 macro en validación, épocas)."""
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    red = Red(g)
    conteo = np.bincount(y, minlength=len(CLASES))
    pesos = torch.tensor(len(y) / (len(CLASES) * np.maximum(conteo, 1)), dtype=torch.float32)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.AdamW(red.parameters(), lr=g["tasa"], weight_decay=g["weight_decay"])
    pasos_por_epoca = math.ceil(len(X) / g["lote"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=g["tasa"], total_steps=epocas * pasos_por_epoca)
    Xt, yt, Xvt = torch.tensor(X), torch.tensor(y), torch.tensor(Xv)
    mejor, estado, sin_mejora, epoca = -1.0, None, 0, 0
    for epoca in range(epocas):
        red.train()
        orden = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(orden), g["lote"]):
            idx = orden[i:i + g["lote"]]
            opt.zero_grad()
            perdida(red(aumentar(Xt[idx], gen, g["aumento"])), yt[idx]).backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 2.0)
            opt.step()
            sched.step()
        f1 = f1_score(yv, probabilidades(red, Xvt).argmax(1), average="macro")
        if not math.isfinite(f1):
            break
        if f1 > mejor + 1e-4:
            mejor, estado, sin_mejora = f1, {k: v.clone() for k, v in red.state_dict().items()}, 0
        else:
            sin_mejora += 1
            if paciencia and sin_mejora >= paciencia:
                break
    if estado:
        red.load_state_dict(estado)
    return red, max(mejor, 0.0), epoca + 1


# ---------------------------------------------------------------- búsqueda

class Evaluador:
    """Aptitud de un genoma con un entrenamiento corto, con caché (un genoma repetido no se vuelve a entrenar)."""

    def __init__(self, X, y, Xv, yv, epocas):
        self.datos, self.epocas, self.cache, self.registro, self.genomas = (X, y, Xv, yv), epocas, {}, [], {}

    def __call__(self, g, etiqueta):
        k = clave(g)
        if k not in self.cache:
            t0 = time.perf_counter()
            _, f1, _ = entrenar(g, *self.datos, epocas=self.epocas)
            self.cache[k] = f1
            self.genomas[k] = dict(g)
            self.registro.append({"busqueda": etiqueta, "evaluacion": len(self.registro) + 1, "f1": round(f1, 4),
                                  "parametros": parametros(g), "segundos": round(time.perf_counter() - t0, 1), **g})
            print(f"    [{etiqueta} {len(self.registro):3d}] F1 {f1:.4f} · {parametros(g):>9,} parámetros · "
                  f"{describir(g)} · {time.perf_counter() - t0:.0f} s", flush=True)
        return self.cache[k]


def describir(g):
    return (f"{g['profundidad']}×{g['ancho']} {g['activacion']}{' res' if g['residual'] else ''}"
            f"{' LN' if g['normalizacion'] else ''} drop {g['dropout']} lr {g['tasa']:g} lote {g['lote']} "
            f"aum {g['aumento']}")


def genetico(evaluar, poblacion_n, generaciones, r):
    poblacion = [genoma_al_azar(r) for _ in range(poblacion_n)]
    curva, aptitud = [], evaluar.cache
    for gen in range(generaciones):
        for g in poblacion:
            evaluar(g, "genético")
        valores = sorted((aptitud[clave(g)] for g in poblacion), reverse=True)
        curva.append({"generacion": gen + 1, "mejor": round(valores[0], 4),
                      "media": round(float(np.mean(valores)), 4),
                      "evaluadas": len(evaluar.cache)})
        print(f"  generación {gen + 1}: mejor {valores[0]:.4f} · media {curva[-1]['media']:.4f} · "
              f"{len(evaluar.cache)} redes entrenadas", flush=True)
        orden = sorted(poblacion, key=lambda g: aptitud[clave(g)], reverse=True)
        nueva = [dict(g) for g in orden[:2]]                                     # elitismo
        while len(nueva) < poblacion_n:
            hijo = mutar(cruzar(torneo(poblacion, aptitud, r), torneo(poblacion, aptitud, r), r), r)
            nueva.append(hijo)
        poblacion = nueva
    # Los mejores de toda la búsqueda (la última población todavía no se evaluó)
    mejores = sorted(evaluar.genomas.values(), key=lambda g: aptitud[clave(g)], reverse=True)
    return curva, mejores


def al_azar(evaluar, presupuesto, r):
    mejor, curva = -1.0, []
    while len(evaluar.cache) < presupuesto:
        f = evaluar(genoma_al_azar(r), "al azar")
        mejor = max(mejor, f)
        curva.append({"evaluadas": len(evaluar.cache), "mejor": round(mejor, 4)})
    return curva


# ---------------------------------------------------------------- exportación

def exportar(red, umbral, reporte):
    """Formato versión 3 de frontend/js/mlp.js: entrada, bloques (con LayerNorm y residual) y salida."""
    def lineal(m):
        return {"W": empaquetar(m.weight.detach().numpy().T), "b": empaquetar(m.bias.detach().numpy())}

    def norma(m):
        return {"g": empaquetar(m.weight.detach().numpy()), "b": empaquetar(m.bias.detach().numpy()),
                "eps": m.eps} if m is not None else None
    g = red.genoma
    return {
        "clases": CLASES, "umbral_confianza": umbral, "version": 3, "activacion": g["activacion"],
        "entrada_capa": lineal(red.entrada),
        "bloques": [{**lineal(b.lineal), "norma": norma(b.norma), "residual": b.residual} for b in red.bloques],
        "salida": {**lineal(red.salida), "norma": norma(red.norma_salida)},
        "genoma": g, "parametros": parametros(g),
        "entrada": "63 valores normalizados (frontend/js/normalize.js: aVector(normalizar(puntos, aspecto)))",
        "origen": "HaGRID (CC BY-SA 4.0) · diseñada por neuroevolución (ml/evolucion/neuroevolucion.py)",
        "fecha": reporte["fecha"],
    }


def red_v2(X):
    """Probabilidades de la red v2 guardada (frontend/ml/mlp_gestos.json, versión 2)."""
    import base64
    modelo = json.loads(MODELO_WEB.read_text(encoding="utf-8"))
    if modelo.get("version") != 2:
        return None, None
    a = X
    for k, c in enumerate(modelo["capas"]):
        W = np.frombuffer(base64.b64decode(c["W"]["float32"]), "<f4").reshape(c["W"]["forma"])
        b = np.frombuffer(base64.b64decode(c["b"]["float32"]), "<f4")
        a = a @ W + b
        if k < len(modelo["capas"]) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), modelo["umbral_confianza"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=ML / "data" / "externo")
    ap.add_argument("--poblacion", type=int, default=14)
    ap.add_argument("--generaciones", type=int, default=10)
    ap.add_argument("--muestra", type=int, default=50000, help="manos de entrenamiento para la aptitud")
    ap.add_argument("--epocas-aptitud", type=int, default=6)
    ap.add_argument("--rapido", action="store_true", help="población y generaciones mínimas (probar el script)")
    args = ap.parse_args()
    if args.rapido:
        args.poblacion, args.generaciones, args.muestra, args.epocas_aptitud = 4, 2, 5000, 1
    torch.set_num_threads(max(1, torch.get_num_threads()))
    r = random.Random(20261028)

    df = preparar.cargar(args.datos)
    parte = df["persona"].map(preparar.particion).to_numpy()
    X = preparar.normalizar(df)
    y = np.array([CLASES.index(c) for c in df["clase"]])
    ent, val, pru = parte == "entrenamiento", parte == "validacion", parte == "prueba"
    rng = np.random.default_rng(0)
    sub = rng.choice(np.flatnonzero(ent), min(args.muestra, int(ent.sum())), replace=False)
    print(f"aptitud: {len(sub)} manos de entrenamiento, {args.epocas_aptitud} épocas, F1 en {val.sum()} de validación")

    t0 = time.perf_counter()
    ev_genetico = Evaluador(X[sub], y[sub], X[val], y[val], args.epocas_aptitud)
    curva, mejores = genetico(ev_genetico, args.poblacion, args.generaciones, r)
    minutos_genetico = (time.perf_counter() - t0) / 60
    n_genetico = len(ev_genetico.registro)

    # Búsqueda al azar con el mismo número de redes entrenadas
    t0 = time.perf_counter()
    ev_azar = Evaluador(X[sub], y[sub], X[val], y[val], args.epocas_aptitud)
    curva_azar = al_azar(ev_azar, n_genetico, random.Random(7))
    mejor_azar = max(ev_azar.cache.values())
    mejor_genetico = max(v for v in ev_genetico.cache.values())
    print(f"\ngenético: mejor F1 {mejor_genetico:.4f} ({n_genetico} redes, {minutos_genetico:.0f} min) · "
          f"al azar: mejor F1 {mejor_azar:.4f} ({len(ev_azar.registro)} redes)")

    # Los 3 mejores genomas, entrenados completos
    finalistas = []
    for g in mejores[:3]:
        t1 = time.perf_counter()
        red, f1, ep = entrenar(g, X[ent], y[ent], X[val], y[val], epocas=4 if args.rapido else 60, paciencia=8)
        finalistas.append({"genoma": g, "f1_validacion": round(f1, 4), "epocas": ep, "red": red,
                           "minutos": round((time.perf_counter() - t1) / 60, 1)})
        print(f"  finalista {describir(g)}: F1 validación {f1:.4f} en {ep} épocas")
    final = max(finalistas, key=lambda f: (f["f1_validacion"], parametros(f["genoma"])))
    red = final["red"]
    umbral, _ = elegir_umbral(probabilidades(red, torch.tensor(X[val])), y[val])

    # Prueba contra la red v2 (la del visor), sobre las mismas manos
    y_pru = np.array(CLASES)[y[pru]]
    pred_evo = np.array(CLASES)[con_umbral(probabilidades(red, torch.tensor(X[pru])), umbral)]
    comparacion = [resumen(f"evolucionada {describir(final['genoma'])}", y_pru, pred_evo)]
    proba_v2, umbral_v2 = red_v2(X[pru])
    f1_v2_val = None
    if proba_v2 is not None:
        pred_v2 = np.array(CLASES)[con_umbral(proba_v2, umbral_v2)]
        comparacion.append(resumen("red v2 [128, 64]", y_pru, pred_v2))
        solo_evo, solo_v2, p = mcnemar(pred_evo == y_pru, pred_v2 == y_pru)
        proba_v2_val, _ = red_v2(X[val])
        f1_v2_val = float(f1_score(y[val], proba_v2_val.argmax(1), average="macro"))
    for fila, pred in zip(comparacion, [pred_evo] + ([pred_v2] if proba_v2 is not None else [])):
        fila["exactitud_ic95"] = [round(v, 4) for v in intervalo_bootstrap(pred == y_pru, repeticiones=2000)[1:]]
    import pandas as pd
    print("\n" + pd.DataFrame(comparacion).to_string(index=False))
    if proba_v2 is not None:
        print(f"McNemar evolucionada vs v2: {solo_evo} / {solo_v2} · p = {p:.2g}")

    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / "neuroevolucion_curva.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["busqueda", "evaluacion", "f1", "parametros", "segundos", *GENES])
        w.writeheader()
        w.writerows(ev_genetico.registro + ev_azar.registro)
    reporte = {
        "fecha": date.today().isoformat(), "dataset": "HaGRID, CC BY-SA 4.0",
        "configuracion": {"poblacion": args.poblacion, "generaciones": args.generaciones, "torneo": 3,
                          "elitismo": 2, "prob_mutacion": 0.25, "manos_aptitud": len(sub),
                          "epocas_aptitud": args.epocas_aptitud, "espacio": GENES},
        "genetico": {"redes_entrenadas": n_genetico, "mejor_f1_validacion_corta": round(mejor_genetico, 4),
                     "minutos": round(minutos_genetico), "curva": curva},
        "al_azar": {"redes_entrenadas": len(ev_azar.registro), "mejor_f1_validacion_corta": round(mejor_azar, 4),
                    "curva": curva_azar},
        "finalistas": [{k: v for k, v in f.items() if k != "red"} for f in finalistas],
        "elegida": {"genoma": final["genoma"], "parametros": parametros(final["genoma"]),
                    "f1_validacion": final["f1_validacion"], "umbral_confianza": umbral},
        "red_v2_f1_validacion": round(f1_v2_val, 4) if f1_v2_val is not None else None,
        "comparacion_prueba": comparacion,
        "mcnemar_prueba": {"solo_evolucionada": solo_evo, "solo_v2": solo_v2, "p": p} if proba_v2 is not None else None,
    }
    if not args.rapido:   # la curva, para la vista de la red (frontend/red.html)
        (RAIZ / "frontend" / "ml" / "neuroevolucion.json").write_text(json.dumps({
            "fecha": reporte["fecha"], "curva": curva, "al_azar": curva_azar,
            "evaluaciones": [{k: e[k] for k in ("busqueda", "evaluacion", "f1", "parametros")} for e in
                             ev_genetico.registro + ev_azar.registro],
            "elegida": reporte["elegida"], "comparacion_prueba": comparacion}, ensure_ascii=False), encoding="utf-8")
    gana = f1_v2_val is None or final["f1_validacion"] > f1_v2_val
    reporte["exportada"] = bool(gana and not args.rapido)
    (RESULTADOS / "neuroevolucion_reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
    if args.rapido or not gana:
        print("(no se exporta: " + ("--rapido" if args.rapido else "no le gana a la v2 en validación") + ")")
        return
    MODELO_WEB.write_text(json.dumps(exportar(red, umbral, reporte), ensure_ascii=False, separators=(",", ":")),
                          encoding="utf-8")
    muestras = rng.choice(np.flatnonzero(pru), 12, replace=False)
    proba = probabilidades(red, torch.tensor(X[muestras]))
    CASOS_TEST.write_text(json.dumps({"clases": CLASES, "casos": [
        {"x": np.round(X[i], 6).tolist(), "proba": np.round(q, 5).tolist()} for i, q in zip(muestras, proba)]}),
        encoding="utf-8")
    print(f"exportada: {MODELO_WEB.relative_to(RAIZ)} ({MODELO_WEB.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
