"""Búsqueda por similitud sobre el catálogo de modelos (sección 5.5).

Cada pieza del catálogo se representa con varias "variantes" de texto: su nombre, sus
sinónimos y su descripción. La consulta se compara con todas y se queda con la mejor
(como en CBR: se recupera el caso más parecido a la consulta).

Método híbrido, elegido con eval/eval_busqueda.py (ver docs/decisiones.md):
  1. Palabras clave: proporción de palabras de un nombre o sinónimo presentes en la consulta.
     Es exacto cuando el usuario usa un nombre conocido y rechaza bien lo que no está.
  2. Embeddings (si el paso 1 no encontró nada): similitud coseno entre vectores de
     sentence-transformers. Resuelve paráfrasis ("máquina que convierte electricidad en movimiento").
Si sentence-transformers no está instalado, se usa solo el paso 1.

La fuente de verdad del catálogo es frontend/models/catalogo.json. Los embeddings se
guardan en backend/catalog/index.json y se recalculan si el catálogo o el modelo cambian.
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from texto import normalizar

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CATALOGO = RAIZ / "frontend" / "models" / "catalogo.json"
RUTA_INDICE = Path(__file__).parent / "catalog" / "index.json"
MODELO_EMBEDDINGS = os.getenv("EMBEDDINGS_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"
UMBRAL_PALABRAS = 0.5
UMBRAL = float(os.getenv("SIMILARITY_THRESHOLD") or 0.62)  # embeddings; calibrado en eval/eval_busqueda.py

RELLENO = {"un", "una", "unos", "unas", "el", "la", "los", "las", "de", "del", "con", "para", "y", "a"}


def _prefijos(modelo):
    """Los modelos E5 se entrenaron con estos prefijos para distinguir consulta de documento."""
    return ("query: ", "passage: ") if "e5" in modelo.lower() else ("", "")


def _variantes(item):
    return [item["nombre"], *item["sinonimos"], item["descripcion"]]


def _palabras(texto):
    ps = [p for p in normalizar(texto).replace(".", " ").replace(",", " ").split() if p not in RELLENO]
    return [p[:-1] if len(p) > 4 and p.endswith("s") else p for p in ps]  # plural simple


class Catalogo:
    def __init__(self, ruta_catalogo=RUTA_CATALOGO, ruta_indice=RUTA_INDICE, usar_embeddings=True,
                 modelo_embeddings=MODELO_EMBEDDINGS):
        self.ruta_catalogo, self.ruta_indice = Path(ruta_catalogo), Path(ruta_indice)
        self.items = json.loads(self.ruta_catalogo.read_text(encoding="utf-8"))
        self.nombre_modelo = modelo_embeddings
        self.prefijo_consulta, self.prefijo_doc = _prefijos(modelo_embeddings)
        self.modelo = self._cargar_modelo(modelo_embeddings) if usar_embeddings else None
        self.metodo = "hibrido" if self.modelo else "palabras"
        self.vectores = self._indice() if self.modelo else None

    # ---------------------------------------------------------------- carga

    @staticmethod
    def _cargar_modelo(nombre):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(nombre)
        except Exception as e:  # no instalado, sin internet en la primera descarga, etc.
            log.warning("Embeddings no disponibles (%s); se usa búsqueda por palabras", e)
            return None

    def _indice(self):
        huella = hashlib.sha256(self.ruta_catalogo.read_bytes()).hexdigest()
        if self.ruta_indice.exists():
            guardado = json.loads(self.ruta_indice.read_text(encoding="utf-8") or "{}")
            if isinstance(guardado, dict) and guardado.get("huella") == huella \
                    and guardado.get("modelo") == self.nombre_modelo:
                return {it["id"]: np.array(it["embeddings"]) for it in guardado["items"]}

        log.info("Calculando embeddings del catálogo (%d piezas)", len(self.items))
        vectores, salida = {}, []
        for item in self.items:
            v = self.modelo.encode([self.prefijo_doc + t for t in _variantes(item)], normalize_embeddings=True)
            vectores[item["id"]] = v
            salida.append({"id": item["id"], "variantes": _variantes(item), "embeddings": v.round(6).tolist()})
        self.ruta_indice.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_indice.write_text(json.dumps({"modelo": self.nombre_modelo, "huella": huella, "items": salida},
                                               ensure_ascii=False), encoding="utf-8")
        return vectores

    # ---------------------------------------------------------------- búsqueda

    def ranking_palabras(self, texto):
        consulta = set(_palabras(texto))
        puntajes = []
        for it in self.items:
            mejor = 0.0
            for variante in _variantes(it)[:-1]:  # la descripción es larga: diluiría el puntaje
                ps = _palabras(variante)
                if ps:
                    mejor = max(mejor, sum(p in consulta for p in ps) / len(ps))
            puntajes.append((it, mejor))
        return sorted(puntajes, key=lambda x: x[1], reverse=True)

    def ranking_embeddings(self, texto):
        q = self.modelo.encode(self.prefijo_consulta + texto, normalize_embeddings=True)
        # vectores normalizados: el producto punto es la similitud coseno
        puntajes = [(it, float(np.max(self.vectores[it["id"]] @ q))) for it in self.items]
        return sorted(puntajes, key=lambda x: x[1], reverse=True)

    def buscar(self, texto, umbral=None):
        """Devuelve la pieza más parecida o None si ninguna supera los umbrales."""
        umbral = UMBRAL if umbral is None else umbral
        item, sim = self.ranking_palabras(texto)[0]
        metodo = "palabras"
        if sim < UMBRAL_PALABRAS:
            if not self.modelo:
                return None
            item, sim = self.ranking_embeddings(texto)[0]
            metodo = "embeddings"
            if sim < umbral:
                return None
        return {"id": item["id"], "nombre": item["nombre"], "modelo_url": item["url"],
                "similitud": round(sim, 3), "metodo": metodo}

    def listar(self):
        return [{k: it[k] for k in ("id", "nombre", "sinonimos", "descripcion", "url")} for it in self.items]


_catalogo = None


def catalogo() -> Catalogo:
    global _catalogo
    if _catalogo is None:
        _catalogo = Catalogo()
    return _catalogo


def buscar_similar(texto, umbral=None):
    return catalogo().buscar(texto, umbral)


def listar_catalogo():
    return catalogo().listar()
