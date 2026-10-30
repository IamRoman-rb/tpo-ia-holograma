"""Búsqueda por similitud sobre el catálogo de modelos (sección 5.5).

Cada pieza del catálogo se representa con varias "variantes" de texto: su nombre, sus
sinónimos y su descripción. La consulta se compara con todas y se queda con la mejor
(como en CBR: se recupera el caso más parecido a la consulta).

Método híbrido, elegido con eval/eval_busqueda.py (ver docs/decisiones.md):
  1. Palabras clave: proporción de palabras de un nombre o sinónimo presentes en la consulta.
     Es exacto cuando el usuario usa un nombre conocido y rechaza bien lo que no está.
  1b. Si no hubo coincidencia, se corrigen los errores de tipeo o de transcripción con la búsqueda difusa
     (busqueda_difusa.py: Damerau-Levenshtein + BK-tree) y se repite el paso 1 ("rulemam" -> "rulemán").
  2. Embeddings (si los pasos anteriores no encontraron nada): similitud coseno entre vectores de
     sentence-transformers. Resuelve paráfrasis ("máquina que convierte electricidad en movimiento").
Si sentence-transformers no está instalado, se usa solo el paso 1.

La fuente de verdad del catálogo es frontend/models/catalogo.json. Los embeddings se
guardan en backend/catalog/index.json y se recalculan si el catálogo o el modelo cambian.

Además está el catálogo de Objaverse (miles de modelos de Sketchfab con licencia abierta, ml/catalogo3d/): su índice
(backend/catalog/objaverse.json + objaverse_indice.npz) se arma con ml/catalogo3d/indexar.py y solo cuentan los modelos
bajados con scripts/descargar_modelos3d.py. Los pasos son los mismos; en cada uno el catálogo propio va primero (sus
modelos están armados para desarmarse) y Objaverse después. Entre modelos de Objaverse igual de parecidos gana el que
tiene más piezas separables y mejor calidad.
"""
import functools
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

import busqueda_difusa
from texto import normalizar

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CATALOGO = RAIZ / "frontend" / "models" / "catalogo.json"
RUTA_INDICE = Path(__file__).parent / "catalog" / "index.json"
RUTA_OBJAVERSE = Path(__file__).parent / "catalog" / "objaverse.json"
RUTA_INDICE_OBJAVERSE = Path(__file__).parent / "catalog" / "objaverse_indice.npz"
# Igual que ml/catalogo3d/comun.py: los datos grandes pueden estar en otro disco
CARPETA_GLBS = Path(os.getenv("HOLOGRAMA_DATOS") or RAIZ / "ml" / "data" / "externo") / "objaverse" / "glbs"
URL_GLBS = "/modelos3d"
MODELO_EMBEDDINGS = os.getenv("EMBEDDINGS_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"
UMBRAL_PALABRAS = 0.5
UMBRAL = float(os.getenv("SIMILARITY_THRESHOLD") or 0.62)  # embeddings; calibrado en eval/eval_busqueda.py

@functools.lru_cache(maxsize=2)
def cargar_embeddings(nombre):
    """Modelo de embeddings de oraciones, cargado una sola vez: lo comparten la búsqueda del catálogo y el
    intérprete de pedidos entrenado (intent_modelo.py)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre)


RELLENO = {"un", "una", "unos", "unas", "el", "la", "los", "las", "de", "del", "con", "para", "y", "a"}


def _prefijos(modelo):
    """Los modelos E5 se entrenaron con estos prefijos para distinguir consulta de documento."""
    return ("query: ", "passage: ") if "e5" in modelo.lower() else ("", "")


def _variantes(item):
    return [item["nombre"], *item["sinonimos"], item["descripcion"]]


def _palabras(texto):
    ps = [p for p in normalizar(texto).replace(".", " ").replace(",", " ").split() if p not in RELLENO]
    return [p[:-1] if len(p) > 4 and p.endswith("s") else p for p in ps]  # plural simple


class CatalogoObjaverse:
    """Los modelos de Objaverse ya bajados, con un índice invertido de palabras y una matriz de embeddings."""

    def __init__(self, ruta_items=RUTA_OBJAVERSE, ruta_indice=RUTA_INDICE_OBJAVERSE, carpeta=CARPETA_GLBS,
                 modelo_embeddings=MODELO_EMBEDDINGS):
        self.ruta_items, self.ruta_indice, self.carpeta = Path(ruta_items), Path(ruta_indice), Path(carpeta)
        self.modelo_embeddings = modelo_embeddings
        self.items = json.loads(self.ruta_items.read_text(encoding="utf-8")) if self.ruta_items.exists() else []
        self.vectores = None
        if self.items and self.ruta_indice.exists():
            indice = np.load(self.ruta_indice)
            if str(indice["modelo"]) == modelo_embeddings:
                self.vectores = indice["vectores"].astype(np.float32)
                self.filas_item, self.filas_texto = indice["filas_item"], indice["filas_texto"]
        self.actualizar()

    def actualizar(self):
        """Vuelve a mirar qué modelos están bajados (después de scripts/descargar_modelos3d.py)."""
        self.bajado = np.array([(self.carpeta / f"{it['id']}.glb").exists() for it in self.items], bool)
        self.variantes, self.por_palabra = [], {}
        for i, it in enumerate(self.items):
            vs = [_palabras(v) for v in [it["nombre"], *it["sinonimos"]]]
            self.variantes.append([v for v in vs if v])
            if self.bajado[i]:
                for p in {p for v in vs for p in v}:
                    self.por_palabra.setdefault(p, []).append(i)
        # desempate entre modelos igual de parecidos: los que se pueden desarmar y los mejores
        self.desempate = np.array([it["calidad"] + (1.5 if (it.get("piezas") or 0) >= 3 else 0) for it in self.items])
        log.info("Objaverse: %d de %d modelos bajados", int(self.bajado.sum()), len(self.items))

    def __len__(self):
        return int(self.bajado.sum()) if self.items else 0

    def palabras(self):
        return set(self.por_palabra)

    def mejor_por_palabras(self, texto):
        """(índice, cobertura del pedido, palabras en común) del mejor modelo cuyo nombre aparece COMPLETO en el
        pedido; entre esos, el nombre más largo y después el desempate. Con miles de nombres, una coincidencia
        parcial casi siempre es otra cosa ("aparato que enfría la comida" -> "aparato de radio")."""
        consulta = set(_palabras(texto))
        mejor, clave_mejor = None, (0, 0.0)
        for i in {i for p in consulta for i in self.por_palabra.get(p, ())}:
            for v in self.variantes[i]:
                n = sum(p in consulta for p in v)
                if n == len(v) and (n, self.desempate[i]) > clave_mejor:
                    mejor, clave_mejor = i, (n, self.desempate[i])
        return mejor, clave_mejor[0] / max(len(consulta), 1), clave_mejor[0]

    def mejor_por_embeddings(self, q):
        """(índice, similitud) del modelo bajado con el texto más parecido a la consulta (vector normalizado)."""
        if self.vectores is None or not len(self):
            return None, -1.0
        similitud = self.vectores @ q
        por_item = np.full(len(self.items), -1.0)
        np.maximum.at(por_item, self.filas_item, similitud[self.filas_texto])
        por_item[~self.bajado] = -1.0
        # entre los que están a 0,02 del mejor, el de mejor desempate
        cerca = np.flatnonzero(por_item >= por_item.max() - 0.02)
        i = int(cerca[np.argmax(self.desempate[cerca])])
        return i, float(por_item[i])

    def resultado(self, i, sim, metodo):
        it = self.items[i]
        return {"id": it["id"], "nombre": it["nombre"], "modelo_url": f"{URL_GLBS}/{it['id']}.glb",
                "similitud": round(sim, 3), "metodo": metodo, "origen": "objaverse", "piezas": it.get("piezas"),
                "atribucion": it["atribucion"]}


class Catalogo:
    def __init__(self, ruta_catalogo=RUTA_CATALOGO, ruta_indice=RUTA_INDICE, usar_embeddings=True,
                 modelo_embeddings=MODELO_EMBEDDINGS, objaverse=None):
        self.ruta_catalogo, self.ruta_indice = Path(ruta_catalogo), Path(ruta_indice)
        self.items = json.loads(self.ruta_catalogo.read_text(encoding="utf-8"))
        self.nombre_modelo = modelo_embeddings
        self.prefijo_consulta, self.prefijo_doc = _prefijos(modelo_embeddings)
        self.modelo = self._cargar_modelo(modelo_embeddings) if usar_embeddings else None
        self.metodo = "hibrido" if self.modelo else "palabras"
        self.vectores = self._indice() if self.modelo else None
        # Sin Objaverse (las pruebas y eval/eval_busqueda.py miden solo el catálogo propio); catalogo() lo suma
        self.objaverse = objaverse

    # ---------------------------------------------------------------- carga

    @staticmethod
    def _cargar_modelo(nombre):
        try:
            return cargar_embeddings(nombre)
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

    @staticmethod
    def _resultado(item, sim, metodo):
        return {"id": item["id"], "nombre": item["nombre"], "modelo_url": item["url"],
                "similitud": round(sim, 3), "metodo": metodo,
                # un modelo generado antes y reutilizado sigue mostrándose como generado por IA
                "origen": item.get("origen", "catalogo")}

    def _por_palabras(self, texto, metodo):
        item, sim = self.ranking_palabras(texto)[0]
        propio = self._resultado(item, sim, metodo) if sim >= UMBRAL_PALABRAS else None
        if self.objaverse:
            i, cobertura, en_comun = self.objaverse.mejor_por_palabras(texto)
            # El catálogo propio va primero; Objaverse gana si nombra más del pedido ("motor de auto" contra el
            # "motor" eléctrico del catálogo). El nombre tiene que cubrir la mitad del pedido: si no, es una
            # paráfrasis y la resuelven los embeddings
            consulta = set(_palabras(texto))
            propias = max((sum(p in consulta for p in _palabras(v)) for v in _variantes(item)[:-1]), default=0)
            if i is not None and cobertura >= UMBRAL_PALABRAS and (not propio or en_comun > propias):
                return self.objaverse.resultado(i, cobertura, metodo)
        return propio

    def buscar(self, texto, umbral=None):
        """Devuelve la pieza más parecida o None si ninguna supera los umbrales."""
        umbral = UMBRAL if umbral is None else umbral
        encontrado, cambios = self._por_palabras(texto, "palabras"), []
        if not encontrado:
            corregido, cambios = self.corrector().corregir(texto)
            if cambios:
                encontrado = self._por_palabras(corregido, "palabras + corrección")
        if not encontrado:
            cambios = []
            if not self.modelo:
                return None
            # Una sola palabra sin coincidencia en el catálogo: los embeddings de palabras sueltas,
            # sin contexto, no son confiables ("mesa" quedaba cerca de "rodamiento"). Las paráfrasis,
            # que es lo que resuelven los embeddings, siempre tienen varias palabras. Se trata como
            # un objeto que no está en el catálogo (ver docs/decisiones.md).
            if len(_palabras(texto)) < 2:
                return None
            item, sim = self.ranking_embeddings(texto)[0]
            encontrado = self._resultado(item, sim, "embeddings")
            if self.objaverse:
                i, sim_o = self.objaverse.mejor_por_embeddings(
                    self.modelo.encode(self.prefijo_consulta + texto, normalize_embeddings=True))
                if sim_o > sim:
                    encontrado, sim = self.objaverse.resultado(i, sim_o, "embeddings"), sim_o
            if sim < umbral:
                return None
        return {**encontrado, **({"correccion": [f"{a} → {b}" for a, b in cambios]} if cambios else {})}

    def corrector(self):
        """El corrector de la búsqueda difusa, con las palabras del catálogo (pesan más que las del vocabulario)."""
        c = busqueda_difusa.corrector()
        nuevas = {p for it in self.items for v in _variantes(it)[:-1] for p in normalizar(v).split()}
        if self.objaverse:
            nuevas |= self.objaverse.palabras()
        c.agregar(sorted(nuevas - set(c.frecuencias)))
        return c

    def recargar(self):
        """Vuelve a leer el catálogo reutilizando el modelo ya cargado (solo recalcula los embeddings)."""
        self.items = json.loads(self.ruta_catalogo.read_text(encoding="utf-8"))
        if self.modelo:
            self.vectores = self._indice()
        if self.objaverse:
            self.objaverse.actualizar()

    def listar(self):
        return [{**{k: it[k] for k in ("id", "nombre", "sinonimos", "descripcion", "url")},
                 "origen": it.get("origen", "catalogo")} for it in self.items]


_catalogo = None


def catalogo() -> Catalogo:
    global _catalogo
    if _catalogo is None:
        _catalogo = Catalogo(objaverse=CatalogoObjaverse())
    return _catalogo


def recargar():
    """Vuelve a leer el catálogo (por ejemplo, después de agregar un modelo generado)."""
    catalogo().recargar()


def buscar_similar(texto, umbral=None):
    return catalogo().buscar(texto, umbral)


def listar_catalogo():
    return catalogo().listar()
