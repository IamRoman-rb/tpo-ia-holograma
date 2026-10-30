"""¿El catálogo 3D de Objaverse devuelve un modelo de lo que se pidió?

Dos sets, sobre los modelos bajados:
  - sinónimos reservados: ml/catalogo3d/seleccionar.py guardó un sinónimo en español de uso común de ~15 % de las
    categorías LVIS sin usarlo para nombrar ningún modelo ("consulta_eval"); se pide el sinónimo solo (al catálogo
    llega el objeto que separó el intérprete, no la frase). Mide si se encuentran nombres que no están escritos;
  - eval/catalogo3d_consultas.csv: objetos escritos a mano, con nombres y con paráfrasis ("máquina para coser ropa").
Acierto = la categoría del modelo devuelto es una de las esperadas, o el modelo se llama exactamente como lo pedido.
También se informa si devolvió algo y la latencia, con intervalos de confianza por bootstrap.

Uso (desde la raíz):  backend/.venv/Scripts/python eval/eval_catalogo3d.py
Salida: eval/resultados/catalogo3d.csv
"""
import csv
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))
import retrieval  # noqa: E402
from estadistica import formato_intervalo, intervalo_bootstrap  # noqa: E402
from texto import normalizar  # noqa: E402

RESULTADOS = RAIZ / "eval" / "resultados"


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cat = retrieval.catalogo()
    if not cat.objaverse:
        sys.exit("No hay modelos de Objaverse bajados: correr scripts/descargar_modelos3d.py")
    categoria = {it["id"]: it["categoria"] for it in cat.objaverse.items}
    nombres = {it["id"]: {normalizar(n) for n in [it["nombre"], *it["sinonimos"]]}
               for it in cat.objaverse.items + cat.items}
    bajadas = {it["categoria"] for it, b in zip(cat.objaverse.items, cat.objaverse.bajado) if b}

    reservados = {}
    for fila in leer(RAIZ / "ml" / "catalogo3d" / "seleccion.csv"):
        if fila["consulta_eval"] and fila["categoria"] in bajadas:
            reservados[fila["categoria"]] = fila["consulta_eval"]
    sets = {
        # Al catálogo llega solo el objeto ("pupitre"), no la frase: lo separa antes el intérprete de pedidos
        "sinonimos_reservados": [{"consulta": s, "categoria": c} for c, s in sorted(reservados.items())],
        # a mano: varias categorías válidas separadas por "|" (una guitarra puede venir del dominio o de LVIS)
        "a_mano": [f for f in leer(RAIZ / "eval" / "catalogo3d_consultas.csv")
                   if set(f["categoria"].split("|")) & bajadas],
    }

    salida = []
    for nombre, filas in sets.items():
        aciertos, devolvio, tiempos = [], [], []
        for f in filas:
            t0 = time.perf_counter()
            r = cat.buscar(f["consulta"])
            tiempos.append((time.perf_counter() - t0) * 1000)
            obtenida = (categoria.get(r["id"]) or f"propio:{r['id']}") if r else None
            # También vale un modelo que se llama exactamente como lo pedido aunque sea de otra categoría LVIS
            # ("pecera" -> la pecera de "fishbowl" y no la de "aquarium"): quien pide no puede distinguirlas
            mismo_nombre = r is not None and normalizar(f["consulta"]) in nombres.get(r["id"], set())
            aciertos.append(obtenida in f["categoria"].split("|") or mismo_nombre)
            devolvio.append(r is not None)
            salida.append({"set": nombre, "consulta": f["consulta"], "esperada": f["categoria"], "obtenida": obtenida,
                           "modelo": r["nombre"] if r else "", "metodo": r["metodo"] if r else "",
                           "similitud": r["similitud"] if r else "", "ok": aciertos[-1], "ms": round(tiempos[-1])})
        print(f"\n== {nombre}: {len(filas)} consultas")
        for s in salida:
            if s["set"] == nombre and not s["ok"]:
                print(f"  FALLA  {s['consulta']!r}: esperaba {s['esperada']}, dio {s['obtenida']} ({s['modelo']})")
        print(f"  categoría correcta: {sum(aciertos)}/{len(filas)} · "
              f"{formato_intervalo(*intervalo_bootstrap(aciertos))} (IC 95 %) · devolvió algo: "
              f"{100 * sum(devolvio) / max(len(filas), 1):.0f} % · latencia mediana {statistics.median(tiempos):.0f} ms")

    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / "catalogo3d.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0]))
        w.writeheader()
        w.writerows(salida)
    print(f"\n{len(cat.objaverse)} modelos bajados · -> eval/resultados/catalogo3d.csv")


if __name__ == "__main__":
    main()
