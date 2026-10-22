"""Mide la recuperación del catálogo contra busqueda_prueba.csv (meta: >= 80 %) y calibra el umbral.

Un acierto es devolver la pieza esperada, o no devolver nada cuando la pieza no está en el
catálogo (en ese caso el producto debería generarla). Compara la búsqueda por palabras, dos
modelos de embeddings y el método híbrido que usa el producto, y barre el umbral de similitud
para elegirlo con datos.

Uso (desde la raíz):  python eval/eval_busqueda.py
La primera vez descarga los modelos de embeddings (cientos de MB).
Ojo: el umbral se calibra sobre estas mismas 20 consultas. Para un número honesto, conviene
medir el umbral elegido sobre un segundo set de consultas que no se haya usado para elegirlo.
"""
import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
sys.path.insert(0, str(RAIZ / "backend"))

from retrieval import MODELO_EMBEDDINGS, UMBRAL, UMBRAL_PALABRAS, Catalogo  # noqa: E402

RESULTADOS = Path(__file__).parent / "resultados"
MODELOS = list(dict.fromkeys([MODELO_EMBEDDINGS, "paraphrase-multilingual-MiniLM-L12-v2",
                              "intfloat/multilingual-e5-small"]))
BARRIDO = [u / 100 for u in range(30, 95, 5)]


def por_ranking(funcion_ranking):
    """Predictor a partir de un ranking: la mejor pieza si supera el umbral."""
    def predecir(consulta, umbral):
        item, sim = funcion_ranking(consulta)[0]
        return (item["id"] if sim >= umbral else ""), item["id"], sim
    return predecir


def por_busqueda(cat):
    def predecir(consulta, umbral):
        r = cat.buscar(consulta, umbral)
        return (r["id"] if r else ""), (r["id"] if r else "-"), (r["similitud"] if r else 0.0)
    return predecir


def evaluar(nombre, etiqueta, predecir, filas, umbral_actual):
    print(f"\n== {nombre}")
    salida, ok = [], 0
    for fila in filas:
        obtenido, top1, sim = predecir(fila["consulta"], umbral_actual)
        ok += obtenido == fila["esperado"]
        if obtenido != fila["esperado"]:
            print(f"  FALLA {fila['consulta']!r} -> {obtenido or 'ninguno'} (mejor: {top1} {sim:.2f}),"
                  f" esperado: {fila['esperado'] or 'ninguno'}")
        salida.append([fila["consulta"], fila["esperado"], obtenido, top1, round(sim, 3)])
    print(f"  Exactitud con el umbral actual ({umbral_actual:.2f}): {100 * ok / len(filas):.0f} %")

    barrido = [(u, sum(predecir(f["consulta"], u)[0] == f["esperado"] for f in filas) / len(filas)) for u in BARRIDO]
    mejor = max(barrido, key=lambda x: x[1])
    print("  Barrido de umbral: " + "  ".join(f"{u:.2f}→{100 * e:.0f}%" for u, e in barrido))
    print(f"  Mejor umbral: {mejor[0]:.2f} ({100 * mejor[1]:.0f} %)")

    with open(RESULTADOS / f"busqueda_{etiqueta}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["consulta", "esperado", "obtenido", "mejor_candidato", "similitud"])
        w.writerows(salida)


def main():
    with open(Path(__file__).parent / "busqueda_prueba.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    RESULTADOS.mkdir(exist_ok=True)

    palabras = Catalogo(usar_embeddings=False)
    evaluar("Palabras clave", "palabras", por_ranking(palabras.ranking_palabras), filas, UMBRAL_PALABRAS)

    for modelo in MODELOS:
        etiqueta = modelo.split("/")[-1]
        cat = Catalogo(ruta_indice=RESULTADOS / f"indice_{etiqueta}.json", modelo_embeddings=modelo)
        if not cat.modelo:
            print(f"\n(embeddings no disponibles con {modelo}: instalá sentence-transformers)")
            continue
        evaluar(f"Embeddings · {modelo}", etiqueta, por_ranking(cat.ranking_embeddings), filas, UMBRAL)
        if modelo == MODELO_EMBEDDINGS:
            evaluar(f"Híbrido (palabras + {modelo}) — el del producto", "hibrido", por_busqueda(cat), filas, UMBRAL)


if __name__ == "__main__":
    main()
