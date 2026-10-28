"""Mide los aciertos de la interpretación de intención contra frases_prueba.csv (meta: >= 85 %).

Compara los tres motores: el LLM, el intérprete entrenado (backend/intent_modelo.py) y las reglas (línea base). Cada salida pasa
por safety.py, igual que en el producto. Un acierto exige que coincidan la acción, la vista,
el factor de escala y, para "mostrar", que el objeto contenga la palabra clave esperada.
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

import intent  # noqa: E402
import intent_modelo  # noqa: E402
from safety import validar_intencion  # noqa: E402
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


def correr(motor, filas, sufijo=""):
    funcion = {"llm": intent.interpretar_llm, "modelo": intent_modelo.interpretar_modelo,
               "reglas": intent.interpretar_reglas}[motor]
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
    print(f"  Aciertos: {aciertos}/{len(filas)} ({100 * aciertos / len(filas):.0f} %)"
          f" · latencia mediana {statistics.median(tiempos):.0f} ms")
    if "objeto" in filas[0]:
        print(f"  Objeto con al menos el núcleo: {nucleo}/{len(filas)} ({100 * nucleo / len(filas):.0f} %)")
    return aciertos / len(filas)


def llm_disponible():
    try:
        intent._cliente_llm().models.list(limit=1)
        return True
    except Exception as e:
        print(f"(LLM no disponible: {type(e).__name__}. Configurá ANTHROPIC_API_KEY en .env para evaluarlo.)")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", choices=["reglas", "modelo", "llm", "todos"], default="todos")
    ap.add_argument("--frases", choices=["prueba", "validacion", "nuevas", "objetos_validacion", "objetos_prueba"], default="prueba")
    args = ap.parse_args()

    with open(Path(__file__).parent / f"frases_{args.frases}.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    motores = ["reglas", "modelo", "llm"] if args.motor == "todos" else [args.motor]
    for motor in motores:
        if motor == "llm" and not llm_disponible():
            continue
        correr(motor, filas, "" if args.frases == "prueba" else f"_{args.frases}")


if __name__ == "__main__":
    main()
