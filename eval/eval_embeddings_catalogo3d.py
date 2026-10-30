"""Elige el modelo de embeddings y el umbral del catálogo 3D de Objaverse.

Compara modelos multilingües con solo el paso de embeddings (sin palabras) sobre:
  - positivos: los sinónimos reservados y las consultas a mano de eval/eval_catalogo3d.py (deben encontrar algo de la
    categoría correcta o con el mismo nombre);
  - negativos: palabras y frases que no son objetos ("felicidad", "qué hora es"): no deben devolver nada.
Para cada modelo y umbral informa aciertos de los positivos y falsos positivos de los negativos, y el tiempo por
consulta. El umbral se elige en este set; la prueba final es eval/eval_catalogo3d.py.

Uso (desde la raíz):  backend/.venv/Scripts/python eval/eval_embeddings_catalogo3d.py [--modelos a,b]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
import retrieval  # noqa: E402
from texto import normalizar  # noqa: E402

MODELOS = ["paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"]
NEGATIVOS = ["felicidad", "democracia", "mañana", "qué hora es", "contame un chiste", "rápido", "el clima de hoy",
             "nada", "algo", "hola", "gracias", "la verdad", "matemática", "ayer llovió mucho", "quiero dormir",
             "silencio", "amistad", "el número siete", "azul", "correr"]
UMBRALES = [round(0.5 + 0.02 * i, 2) for i in range(21)]


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=",".join(MODELOS))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = json.loads(retrieval.RUTA_OBJAVERSE.read_text(encoding="utf-8"))
    bajado = np.array([(retrieval.CARPETA_GLBS / f"{it['id']}.glb").exists() for it in items])
    nombres = [{normalizar(n) for n in [it["nombre"], *it["sinonimos"]]} for it in items]
    seleccion = leer(RAIZ / "ml" / "catalogo3d" / "seleccion.csv")
    descripcion = {x["uid"]: x["descripcion_en"][:300] for x in seleccion}
    bajadas = {it["categoria"] for it, b in zip(items, bajado) if b}
    positivos = sorted({(x["consulta_eval"], x["categoria"]) for x in seleccion
                        if x["consulta_eval"] and x["categoria"] in bajadas})
    positivos += [(f["consulta"], f["categoria"]) for f in leer(RAIZ / "eval" / "catalogo3d_consultas.csv")]

    textos, filas_item = [], []
    for i, it in enumerate(items):
        if bajado[i]:
            for t in dict.fromkeys([it["nombre"], *it["sinonimos"], descripcion.get(it["id"], "")]):
                if t:
                    textos.append(t)
                    filas_item.append(i)
    filas_item = np.array(filas_item)
    print(f"{int(bajado.sum())} modelos bajados · {len(textos)} textos · {len(positivos)} positivos · "
          f"{len(NEGATIVOS)} negativos")

    for nombre in args.modelos.split(","):
        modelo = retrieval.cargar_embeddings(nombre)
        p_consulta, p_doc = retrieval._prefijos(nombre)
        t0 = time.perf_counter()
        V = modelo.encode([p_doc + t for t in textos], batch_size=64, normalize_embeddings=True)
        minutos = (time.perf_counter() - t0) / 60

        def mejor(consulta):
            s = V @ modelo.encode(p_consulta + consulta, normalize_embeddings=True)
            k = int(np.argmax(s))
            return int(filas_item[k]), float(s[k])

        t0 = time.perf_counter()
        pos = []
        for consulta, cats in positivos:
            i, s = mejor(consulta)
            ok = items[i]["categoria"] in cats.split("|") or normalizar(consulta) in nombres[i]
            pos.append((ok, s))
        ms = (time.perf_counter() - t0) * 1000 / len(positivos)
        neg = [mejor(c)[1] for c in NEGATIVOS]
        print(f"\n== {nombre} · índice {minutos:.1f} min · {ms:.0f} ms por consulta")
        print(f"  top-1 correcto sin umbral: {sum(o for o, _ in pos)}/{len(pos)}")
        for u in UMBRALES:
            aciertos = sum(o and s >= u for o, s in pos)
            equivocados = sum((not o) and s >= u for o, s in pos)
            falsos = sum(s >= u for s in neg)
            print(f"  umbral {u:.2f}: aciertos {aciertos}/{len(pos)} · devuelve otra cosa {equivocados} · "
                  f"negativos aceptados {falsos}/{len(neg)}")


if __name__ == "__main__":
    main()
