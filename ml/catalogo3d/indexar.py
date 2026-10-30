"""Índice de búsqueda del catálogo 3D de Objaverse (backend/catalog/objaverse.json + objaverse_indice.npz).

Para cada modelo de ml/catalogo3d/seleccion.csv:
  - nombre y sinónimos en español (los busca el ranking por palabras de backend/retrieval.py);
  - embeddings de los nombres en español y la descripción de Cap3D en inglés, con el mismo modelo
    de embeddings que el resto del catálogo (multilingüe: una consulta en español encuentra un texto en inglés);
  - cuántas piezas separables tiene (nodos con malla del .glb), si ya está bajado: entre modelos parecidos se prefiere
    el que se puede desarmar.
Los textos repetidos (los nombres de una categoría) se codifican una sola vez.

Uso (desde la raíz, después de scripts/descargar_modelos3d.py):  backend/.venv/Scripts/python ml/catalogo3d/indexar.py
"""
import csv
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import GLBS, RAIZ, SELECCION  # noqa: E402

sys.path.insert(0, str(RAIZ / "backend"))
import retrieval  # noqa: E402

SALIDA_ITEMS = RAIZ / "backend" / "catalog" / "objaverse.json"
SALIDA_INDICE = RAIZ / "backend" / "catalog" / "objaverse_indice.npz"
LARGO_DESCRIPCION = 300


def piezas_glb(ruta):
    """Nodos con malla del .glb (lee solo el bloque JSON del encabezado). None si no está o no se puede leer."""
    try:
        with open(ruta, "rb") as f:
            magia, _, _ = struct.unpack("<4sII", f.read(12))
            largo, tipo = struct.unpack("<I4s", f.read(8))
            if magia != b"glTF" or tipo != b"JSON":
                return None
            gltf = json.loads(f.read(largo))
        return sum(1 for n in gltf.get("nodes", []) if "mesh" in n) or None
    except (OSError, struct.error, ValueError):
        return None


def main():
    with open(SELECCION, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    t0 = time.perf_counter()
    items, textos, filas_item, filas_texto = [], {}, [], []
    for i, x in enumerate(filas):
        nombres = [n for n in x["nombres_es"].split("|") if n]
        nombre = (nombres[0] if nombres else x["nombre_en"]).strip()
        items.append({
            "id": x["uid"], "nombre": nombre[:1].upper() + nombre[1:], "sinonimos": nombres[1:],
            "nombre_original": x["nombre_en"], "descripcion": x["descripcion_en"][:LARGO_DESCRIPCION],
            "categoria": x["categoria"], "piezas": piezas_glb(GLBS / f"{x['uid']}.glb"),
            "calidad": float(x["calidad"]), "tamano": int(x["tamano"]),
            "atribucion": {"titulo": x["nombre_en"], "autor": x["autor"], "autor_url": x["autor_url"],
                           "licencia": x["licencia"], "url": x["url"]},
        })
        # El nombre original no se embebe: muchos son genéricos ("e-Commerce 3D Model", "Free Samples") y quedaban
        # cerca de cualquier pedido. La descripción de Cap3D dice lo que el modelo muestra.
        for t in dict.fromkeys(nombres + [x["descripcion_en"][:LARGO_DESCRIPCION]]):
            if t.strip():
                filas_item.append(i)
                filas_texto.append(textos.setdefault(t.strip(), len(textos)))
    print(f"{len(items):,} modelos · {len(textos):,} textos distintos · "
          f"{sum(1 for it in items if it['piezas']):,} bajados", flush=True)

    modelo = retrieval.cargar_embeddings(retrieval.MODELO_EMBEDDINGS)
    _, prefijo_doc = retrieval._prefijos(retrieval.MODELO_EMBEDDINGS)
    vectores = modelo.encode([prefijo_doc + t for t in textos], batch_size=128, normalize_embeddings=True,
                             show_progress_bar=False)
    SALIDA_ITEMS.write_text(json.dumps(items, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(SALIDA_INDICE, vectores=vectores.astype(np.float16),
                        filas_item=np.array(filas_item, np.int32), filas_texto=np.array(filas_texto, np.int32),
                        modelo=np.array(retrieval.MODELO_EMBEDDINGS),
                        huella=np.array(hashlib.sha256(SALIDA_ITEMS.read_bytes()).hexdigest()))
    print(f"-> {SALIDA_ITEMS.relative_to(RAIZ)} ({SALIDA_ITEMS.stat().st_size // 2**20} MB) · "
          f"{SALIDA_INDICE.relative_to(RAIZ)} ({SALIDA_INDICE.stat().st_size // 2**20} MB) · "
          f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
