"""Mide los aciertos de la interpretación de intención contra frases_prueba.csv (meta: >= 85 %).

Compara los motores: el LLM (Claude), el consenso de los motores propios (backend/consenso.py), el LLM propio
ajustado (backend/llm_propio.py), el transformer ajustado (backend/intent_transformer.py), el intérprete con
embeddings (backend/intent_modelo.py), Naive Bayes (backend/intent_bayes.py) y las reglas (línea base).
Cada exactitud va con su intervalo del 95 % (bootstrap) y cada par de motores con el test de McNemar. Cada salida pasa
por safety.py, igual que en el producto. Un acierto exige que coincidan la acción, la vista,
el factor de escala y, para "mostrar", que el objeto contenga la palabra clave esperada.
Los sets de comandos (frases_comandos_*.csv: "usá el árbol para los gestos", "¿qué modelo respondió?") piden la
acción y el motor, el ámbito o la pregunta.
Los sets de objetos (frases_objetos_*.csv, objetos de todo tipo) piden la acción y el objeto EXACTO, con todo lo
que lo describe ("silla de oficina con ruedas"); además se informa cuántas veces el objeto trae al menos el núcleo.

Uso (desde la raíz):
    python eval/eval_intent.py                 # ambos motores (el LLM solo si hay credenciales)
    python eval/eval_intent.py --motor reglas
    python eval/eval_intent.py --motor modelo --frases validacion   # el set con el que se eligió el modelo
    python eval/eval_intent.py --frases objetos_prueba              # objetos de todo tipo, nunca vistos
Cada corrida del LLM hace 30 llamadas a la API (tiene costo).
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

import intent  # noqa: E402
import consenso  # noqa: E402
import intent_bayes  # noqa: E402
import intent_transformer  # noqa: E402
import intent_modelo  # noqa: E402
import llm_propio  # noqa: E402
from safety import validar_intencion  # noqa: E402
from estadistica import formato_intervalo, intervalo_bootstrap, mcnemar  # noqa: E402
from texto import normalizar  # noqa: E402

RESULTADOS = Path(__file__).parent / "resultados"


def factor_ok(esperado: str, obtenido):
    if not esperado:
        return obtenido is None
    if obtenido is None:
        return False
    if esperado == ">1":
        return obtenido > 1
    if esperado == "<1":
        return obtenido < 1
    return abs(obtenido - float(esperado)) <= 0.05 * float(esperado)


def evaluar_fila(fila, r):
    errores = []
    if "pregunta" in fila:  # sets de comandos: la acción y, si es un comando, sus parámetros
        if r["accion"] != fila["accion"]:
            errores.append(f"acción {r['accion']}")
        elif fila["accion"] in ("usar_motor", "consultar"):
            for clave in ("motor", "ambito", "pregunta"):
                if (r["parametros"].get(clave) or "") != fila[clave]:
                    errores.append(f"{clave} {r['parametros'].get(clave)}")
        return errores
    if "objeto" in fila:   # sets de objetos: acción y objeto exacto
        if r["accion"] != fila["accion"]:
            errores.append(f"acción {r['accion']}")
        if normalizar(r["objeto"] or "") != normalizar(fila["objeto"]):
            errores.append(f"objeto {r['objeto']!r}")
        return errores
    if r["accion"] != fila["accion"]:
        errores.append(f"acción {r['accion']}")
    if (r["parametros"]["vista"] or "") != fila["vista"]:
        errores.append(f"vista {r['parametros']['vista']}")
    if not factor_ok(fila["factor_escala"], r["parametros"]["factor_escala"]):
        errores.append(f"escala {r['parametros']['factor_escala']}")
    if fila["accion"] in ("mostrar", "aislar") and fila["objeto_clave"] not in normalizar(r["objeto"] or ""):
        errores.append(f"objeto {r['objeto']!r}")
    return errores


def _llm_propio(frase):
    try:
        return llm_propio.interpretar_llm_propio(frase)
    except llm_propio.RespuestaInvalida:   # escribió algo que no es una acción: cuenta como falla
        return dict(intent.DESCONOCIDO)


DISPONIBLE = {
    "llm_propio": lambda: (llm_propio.RUTA_MODELO / "config.json").exists(),
    "transformer": lambda: (intent_transformer.RUTA_MODELO / "interprete.json").exists(),
    "bayes": lambda: intent_bayes.RUTA_MODELO.exists(),
}


def correr(motor, filas, sufijo=""):
    funcion = {"llm": intent.interpretar_llm, "modelo": intent_modelo.interpretar_modelo,
               "llm_propio": _llm_propio, "bayes": intent_bayes.interpretar_bayes,
               "transformer": intent_transformer.interpretar_transformer,
               "consenso": consenso.interpretar_consenso, "reglas": intent.interpretar_reglas}[motor]
    salida, aciertos, tiempos, nucleo = [], 0, [], 0
    for fila in filas:
        t0 = time.perf_counter()
        r = validar_intencion(funcion(fila["frase"]))
        tiempos.append((time.perf_counter() - t0) * 1000)
        errores = evaluar_fila(fila, r)
        aciertos += not errores
        if "objeto" in fila:
            nucleo += normalizar(fila["objeto"]).split()[0] in normalizar(r["objeto"] or "").split()
        salida.append({"frase": fila["frase"], "ok": not errores, "errores": "; ".join(errores),
                       "accion": r["accion"], "objeto": r["objeto"], "vista": r["parametros"]["vista"],
                       "factor_escala": r["parametros"]["factor_escala"], "confianza": r["confianza"],
                       "ms": round(tiempos[-1])})

    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / f"intent_{motor}{sufijo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0]))
        w.writeheader()
        w.writerows(salida)

    print(f"\n== Motor: {motor}")
    for s in salida:
        if not s["ok"]:
            print(f"  FALLA  {s['frase']!r}: {s['errores']}")
    print(f"  Aciertos: {aciertos}/{len(filas)} · {formato_intervalo(*intervalo_bootstrap([s['ok'] for s in salida]))}"
          f" (IC 95 % bootstrap) · latencia mediana {statistics.median(tiempos):.0f} ms")
    if "objeto" in filas[0]:
        print(f"  Objeto con al menos el núcleo: {nucleo}/{len(filas)} ({100 * nucleo / len(filas):.0f} %)")
    return [s["ok"] for s in salida]


def llm_disponible():
    try:
        intent._cliente_llm().models.list(limit=1)
        return True
    except Exception as e:
        print(f"(LLM no disponible: {type(e).__name__}. Configurá ANTHROPIC_API_KEY en .env para evaluarlo.)")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", choices=["reglas", "bayes", "modelo", "transformer", "llm_propio", "consenso", "llm", "todos"], default="todos")
    ap.add_argument("--frases", choices=["prueba", "validacion", "nuevas", "objetos_validacion", "objetos_prueba",
                                                 "comandos_validacion", "comandos_prueba"], default="prueba")
    args = ap.parse_args()

    with open(Path(__file__).parent / f"frases_{args.frases}.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    motores = ["reglas", "bayes", "modelo", "transformer", "llm_propio", "consenso", "llm"] if args.motor == "todos" else [args.motor]
    resultados = {}
    for motor in motores:
        if motor == "llm" and not llm_disponible():
            continue
        if motor in DISPONIBLE and not DISPONIBLE[motor]():
            print(f"({motor} no entrenado: ver ml/README.md para entrenarlo.)")
            continue
        resultados[motor] = correr(motor, filas, "" if args.frases == "prueba" else f"_{args.frases}")

    # ¿Las diferencias son significativas? McNemar sobre las mismas frases
    nombres = list(resultados)
    if len(nombres) > 1:
        print("\n== McNemar (solo A / solo B · p; p < 0,05: la diferencia no parece casualidad)")
        for i, a in enumerate(nombres):
            for b in nombres[i + 1:]:
                solo_a, solo_b, p = mcnemar(resultados[a], resultados[b])
                print(f"  {a:>10s} vs {b:<10s} {solo_a:3d} / {solo_b:<3d} · p = {p:.3f}{' *' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
