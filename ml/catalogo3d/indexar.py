"""Índice de búsqueda del catálogo 3D de Objaverse (backend/catalog/objaverse.json + objaverse_indice.npz).

Para cada modelo de ml/catalogo3d/seleccion.csv:
  - nombre y sinónimos en español (los busca el ranking por palabras de backend/retrieval.py);
  - embeddings de los nombres en español y la descripción de Cap3D en inglés, con el mismo modelo
    de embeddings que el resto del catálogo (multilingüe: una consulta en español encuentra un texto en inglés);
  - cuántas piezas separables tiene (las que ve el visor: el primer nivel del árbol de nodos con dos o más hijos con
    mallas, hasta 80), si ya está bajado: entre modelos parecidos se prefiere el que se puede desarmar.
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


# Igual que frontend/js/piezas.js (nivelDePiezas): más de 80 piezas, el visor las agrupa por cercanía
MAXIMO_PIEZAS = 80
CUOTA_DOMINANTE = 0.9


def contar_piezas(gltf):
    """Piezas que ve el visor. Los modelos de Sketchfab tienen un único nodo raíz y las mallas cuelgan varios niveles
    más abajo: desde las raíces de la escena, mientras un nodo sin malla propia concentre más del 90 % de las mallas
    del nivel (por ejemplo, el único hijo de una cadena), se lo reemplaza por sus hijos."""
    nodos = gltf.get("nodes", [])
    mallas = {}

    def contar(i):  # mallas del nodo y sus descendientes (un .glb con ciclos termina en RecursionError)
        if i not in mallas:
            mallas[i] = ("mesh" in nodos[i]) + sum(contar(h) for h in nodos[i].get("children", []))
        return mallas[i]

    def con_mallas(ids):
        return [i for i in ids if contar(i) > 0]

    escenas = gltf.get("scenes", [])
    if not escenas:
        return 0
    nivel = con_mallas(escenas[gltf.get("scene", 0)].get("nodes", []))
    while True:
        total = sum(contar(i) for i in nivel)
        k = next((k for k, i in enumerate(nivel) if "mesh" not in nodos[i] and contar(i) > CUOTA_DOMINANTE * total),
                 None)
        if k is None:
            return min(len(nivel), MAXIMO_PIEZAS)
        nivel = nivel[:k] + con_mallas(nodos[nivel[k]].get("children", [])) + nivel[k + 1:]


def piezas_glb(ruta):
    """Piezas separables que ve el visor (contar_piezas; lee solo el bloque JSON del encabezado).
    None si no está, no se puede leer o no tiene mallas."""
    try:
        with open(ruta, "rb") as f:
            magia, _, _ = struct.unpack("<4sII", f.read(12))
            largo, tipo = struct.unpack("<I4s", f.read(8))
            if magia != b"glTF" or tipo != b"JSON":
                return None
            gltf = json.loads(f.read(largo))
        return contar_piezas(gltf) or None
    except (OSError, struct.error, ValueError, IndexError, KeyError, RecursionError):
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
