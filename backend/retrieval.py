"""Búsqueda por similitud (embeddings + coseno) sobre el catálogo de modelos."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RUTA_CATALOGO = Path(__file__).parent / "catalog" / "index.json"

modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def cargar_catalogo(ruta=RUTA_CATALOGO):
    with open(ruta, encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        it["vec"] = np.array(it["embedding"])
    return items


CATALOGO = cargar_catalogo()


def listar_catalogo():
    return [{k: v for k, v in it.items() if k not in ("embedding", "vec")} for it in CATALOGO]


def buscar_similar(texto, umbral=0.75):
    q = modelo.encode(texto)
    mejor, mejor_sim = None, -1
    for it in CATALOGO:
        sim = float(np.dot(q, it["vec"]) /
                    (np.linalg.norm(q) * np.linalg.norm(it["vec"])))
        if sim > mejor_sim:
            mejor, mejor_sim = it, sim
    if mejor_sim >= umbral:
        return {"modelo_url": mejor["url"], "similitud": round(mejor_sim, 3)}
    return None
