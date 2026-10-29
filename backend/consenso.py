"""Consenso de motores (motor "consenso"): la mejor respuesta, calculada con matemática a partir de todos.

Cada motor propio (el LLM propio, el transformer, el intérprete con embeddings, Naive Bayes y las reglas)
interpreta la frase en paralelo y da una distribución de probabilidad sobre las acciones. Se combinan así:
  1. Acción: combinación log-lineal (un "producto de expertos" ponderado):
        log P(a | frase) = Σ_m  w_m · log p_m(a) + b_a  − log Z
     Los pesos w_m (cuánto confiar en cada motor) y los sesgos b_a se ajustan en validación maximizando la
     log-verosimilitud con L-BFGS (ml/consenso/ajustar_consenso.py). Un motor que se equivoca seguido termina
     con peso chico; uno muy seguro y certero, con peso grande.
  2. Objeto: voto ponderado entre los objetos que propusieron los motores que eligieron esa acción; los que
     dicen lo mismo con otras palabras ("silla de oficina" y "la silla de oficina") se agrupan.
  3. Decisión bayesiana: con P(acierto) = P(acción elegida), ejecutar cuesta en promedio (1 − P) × 1 (equivocarse)
     y pedir confirmación cuesta siempre C (molestar al usuario, 0,2). Conviene preguntar cuando
     (1 − P) > C, es decir P < 1 − C. El umbral sale de los costos, no a ojo.
Si un motor no está instalado o falla, el consenso sigue con los demás.
"""
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from texto import normalizar

log = logging.getLogger(__name__)

RUTA_PESOS = Path(__file__).parent / "modelos" / "consenso.json"
COSTO_PREGUNTA = 0.2      # costo de pedir confirmación, relativo a equivocarse (= 1)
PISO = 1e-4               # probabilidad mínima de una acción (un motor nunca veta del todo)
CONFIANZA_REGLAS = 0.7


def acciones():
    from schemas import Accion
    from typing import get_args
    return list(get_args(Accion))


# ---------------------------------------------------------------- cada motor, con su distribución

def _reglas(frase, traza):
    import intent
    return intent.interpretar_reglas(frase)    # sin distribución: se arma con la confianza fija


def _bayes(frase, traza):
    import intent_bayes
    return intent_bayes.interpretar_bayes(frase, traza=traza)


def _modelo(frase, traza):
    import intent_modelo
    return intent_modelo.interpretar_modelo(frase, traza=traza)


def _transformer(frase, traza):
    import intent_transformer
    return intent_transformer.interpretar_transformer(frase, traza=traza)


def _llm_propio(frase, traza):
    import llm_propio
    return llm_propio.interpretar_llm_propio(frase, traza=traza, distribucion=True)


# Lo que cada motor cuenta de cómo decidió (lo muestra la vista del LLM, llm.html)
DETALLE = ("crudo_llm_propio", "tokens_entrada", "tokens_salida", "llm_propio", "rasgos_decisivos", "bio",
           "palabras_objeto", "transformer")


MOTORES = {"llm_propio": _llm_propio, "transformer": _transformer, "modelo": _modelo, "bayes": _bayes,
           "reglas": _reglas}


def distribucion_completa(r, dist, lista):
    """Distribución sobre todas las acciones. Sin una del motor (reglas), la confianza va a su acción."""
    if dist:
        p = np.array([dist.get(a, 0.0) for a in lista], float)
    else:
        resto = (1 - CONFIANZA_REGLAS) / (len(lista) - 1)
        p = np.array([CONFIANZA_REGLAS if a == r["accion"] else resto for a in lista], float)
    p = np.maximum(p, PISO)
    return p / p.sum()


def correr_motores(frase, motores=None):
    """Corre los motores en paralelo. Devuelve {motor: {intencion, distribucion, ms}} con los que respondieron."""
    motores = motores or list(MOTORES)
    lista = acciones()

    def uno(nombre):
        t0 = time.perf_counter()
        traza = {}
        try:
            r = MOTORES[nombre](frase, traza)
        except Exception as e:   # no instalado, sin pesos o una respuesta inválida: ese motor no vota
            log.info("consenso: %s no disponible (%s)", nombre, type(e).__name__)
            return nombre, None
        return nombre, {"intencion": r, "distribucion": distribucion_completa(r, traza.get("distribucion"), lista),
                        "ms": round((time.perf_counter() - t0) * 1000),
                        "detalle": {k: traza[k] for k in DETALLE if k in traza}}

    with ThreadPoolExecutor(max_workers=len(motores)) as ex:
        return {n: v for n, v in ex.map(uno, motores) if v is not None}


# ---------------------------------------------------------------- combinación

@lru_cache(maxsize=1)
def pesos_guardados():
    if RUTA_PESOS.exists():
        return json.loads(RUTA_PESOS.read_text(encoding="utf-8"))
    return None


def combinar(salidas, pesos=None, lista=None):
    """Distribución combinada (log-lineal). pesos: {"pesos": {motor: w}, "sesgos": {accion: b}}."""
    lista = lista or acciones()
    pesos = pesos or pesos_guardados() or {}
    w = pesos.get("pesos", {})
    sesgos = pesos.get("sesgos", {})
    z = np.array([sesgos.get(a, 0.0) for a in lista], float)
    for nombre, s in salidas.items():
        z += w.get(nombre, 1.0 / max(1, len(salidas))) * np.log(s["distribucion"])
    p = np.exp(z - z.max())
    return p / p.sum()


def _similares(a, b):
    """Dos objetos dicen lo mismo si comparten casi todas las palabras (Jaccard >= 0,6)."""
    pa, pb = set(normalizar(a).split()) - {"el", "la", "los", "las", "un", "una"}, \
        set(normalizar(b).split()) - {"el", "la", "los", "las", "un", "una"}
    return bool(pa | pb) and len(pa & pb) / len(pa | pb) >= 0.6


def votar_objeto(salidas, accion, pesos=None):
    """El objeto más votado entre los motores que eligieron esa acción, ponderado por peso × confianza."""
    w = (pesos or pesos_guardados() or {}).get("pesos", {})
    lista = acciones()
    k = lista.index(accion)
    grupos = []   # [puntaje, [(puntaje, objeto)]]
    for nombre, s in salidas.items():
        r = s["intencion"]
        if r["accion"] != accion or not r.get("objeto"):
            continue
        voto = w.get(nombre, 1.0) * float(s["distribucion"][k])
        grupo = next((g for g in grupos if _similares(g[1][0][1], r["objeto"])), None)
        if grupo:
            grupo[0] += voto
            grupo[1].append((voto, r["objeto"]))
        else:
            grupos.append([voto, [(voto, r["objeto"])]])
    if not grupos:
        return None
    mejor = max(grupos, key=lambda g: g[0])
    return max(mejor[1])[1]


def interpretar_consenso(frase, traza=None, motores=None):
    """Misma salida que intent.interpretar_llm, más "umbral_confirmacion" (la decisión bayesiana)."""
    import intent_modelo
    lista = acciones()
    t0 = time.perf_counter()
    salidas = correr_motores(frase, motores)
    if not salidas:
        raise RuntimeError("ningún motor respondió")
    p = combinar(salidas, lista=lista)
    k = int(np.argmax(p))
    accion, confianza = lista[k], float(p[k])
    objeto = votar_objeto(salidas, accion) if accion in intent_modelo.ACCIONES_CON_OBJETO else None
    r = intent_modelo.completar(frase, accion, objeto, round(confianza, 3))
    # Los parámetros de los comandos y la vista los sacan las reglas dentro de completar; si algún motor que
    # eligió la misma acción trae parámetros más completos (Claude no participa), se respetan los de completar.
    umbral = 1 - COSTO_PREGUNTA
    costo_ejecutar = 1 - confianza
    decision = (f"ejecutar: el costo esperado de equivocarse ({costo_ejecutar:.2f}) es menor que el de preguntar "
                f"({COSTO_PREGUNTA:.1f})" if costo_ejecutar <= COSTO_PREGUNTA else
                f"preguntar: el costo esperado de equivocarse ({costo_ejecutar:.2f}) supera al de preguntar "
                f"({COSTO_PREGUNTA:.1f})")
    if traza is not None:
        w = (pesos_guardados() or {}).get("pesos", {})
        orden = np.argsort(p)[::-1][:3]
        traza.update(
            votos=[{"motor": n, "accion": s["intencion"]["accion"], "objeto": s["intencion"].get("objeto"),
                    "confianza": round(float(s["distribucion"].max()), 3), "peso": round(w.get(n, 1 / len(salidas)), 3),
                    "ms": s["ms"], "distribucion": [round(float(v), 4) for v in s["distribucion"]],
                    **s.get("detalle", {})} for n, s in salidas.items()],
            acciones=lista, distribucion=[round(float(v), 4) for v in p],
            sesgos=(pesos_guardados() or {}).get("sesgos", {}), costo_pregunta=COSTO_PREGUNTA,
            probabilidades={lista[i]: round(float(p[i]), 3) for i in orden},
            decision=decision, ms_consenso=round((time.perf_counter() - t0) * 1000),
            pesos_ajustados=pesos_guardados() is not None)
    if accion == "desconocido":
        return r
    return {**r, "umbral_confirmacion": umbral}


def log_verosimilitud(salidas_por_frase, etiquetas, pesos):
    """Para el ajuste: suma de log P(acción correcta) con esos pesos (ml/consenso/ajustar_consenso.py)."""
    lista = acciones()
    total = 0.0
    for salidas, a in zip(salidas_por_frase, etiquetas):
        total += math.log(max(combinar(salidas, pesos, lista)[lista.index(a)], 1e-12))
    return total
