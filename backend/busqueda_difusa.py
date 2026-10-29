"""Búsqueda difusa de palabras: corrige errores de tipeo y de transcripción ("rulemam" -> "rulemán").

- Distancia de edición de Damerau-Levenshtein (versión "alineamiento óptimo"): cuántas letras hay que
  insertar, borrar, cambiar o permutar entre vecinas para pasar de una palabra a otra. Se calcula con
  programación dinámica en O(n·m).
- BK-tree (Burkhard-Keller): un árbol donde cada hijo cuelga a la distancia que tiene con su padre. Por la
  desigualdad triangular, para buscar las palabras a distancia <= k de la consulta solo hay que bajar por los
  hijos a distancia entre d-k y d+k: se revisa una pequeña parte del vocabulario en lugar de las 15 mil palabras.

El vocabulario (backend/modelos/vocabulario.json) son las palabras de los nombres de WordNet en español, de los
objetos industriales y de las frases de entrenamiento, con su frecuencia (ml/intencion/generar_frases.py lo
escribe), más las del catálogo. Una palabra conocida nunca se corrige. Para saber si una palabra existe aunque
no esté en el vocabulario ("compresor"), se usa la frecuencia del español de wordfreq (Zipf: 3 = una vez cada
millón de palabras). Una desconocida se cambia por la candidata más probable según un modelo de canal ruidoso:
P(palabra) · P(error | palabra), con P(palabra) de wordfreq y un costo fijo por cada letra mal escrita, dentro
de una tolerancia que crece con el largo (las palabras cortas se parecen demasiado entre sí).
"""
import json
import math
from functools import lru_cache
from pathlib import Path

from texto import normalizar

RUTA_VOCABULARIO = Path(__file__).parent / "modelos" / "vocabulario.json"
ZIPF_CONOCIDA = 2.0      # desde acá una palabra del español se considera bien escrita (sin wordfreq: solo el vocabulario)
COSTO_POR_ERROR = 2.0    # log10: cada letra mal escrita hace 100 veces menos probable a la candidata


def _zipf():
    try:
        from wordfreq import zipf_frequency
        return lambda p: zipf_frequency(p, "es")
    except ImportError:
        return None


def distancia(a, b, maximo=None):
    """Damerau-Levenshtein (alineamiento óptimo). Con maximo, corta antes si la distancia ya lo supera."""
    if a == b:
        return 0
    if maximo is not None and abs(len(a) - len(b)) > maximo:
        return maximo + 1
    anterior2, anterior = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        actual = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            costo = a[i - 1] != b[j - 1]
            actual[j] = min(anterior[j] + 1, actual[j - 1] + 1, anterior[j - 1] + costo)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                actual[j] = min(actual[j], anterior2[j - 2] + 1)    # dos letras vecinas permutadas
        if maximo is not None and min(actual) > maximo:
            return maximo + 1
        anterior2, anterior = anterior, actual
    return anterior[-1]


class BKTree:
    def __init__(self, palabras=()):
        self.raiz = None   # (palabra, {distancia: hijo})
        self.tamano = 0
        for p in palabras:
            self.agregar(p)

    def agregar(self, palabra):
        if self.raiz is None:
            self.raiz, self.tamano = (palabra, {}), 1
            return
        nodo = self.raiz
        while True:
            d = distancia(palabra, nodo[0])
            if d == 0:
                return
            if d not in nodo[1]:
                nodo[1][d] = (palabra, {})
                self.tamano += 1
                return
            nodo = nodo[1][d]

    def buscar(self, palabra, tolerancia):
        """Palabras a distancia <= tolerancia, como (distancia, palabra), de la más cercana a la más lejana.
        También devuelve cuántas distancias calculó (para mostrar cuánto poda el árbol)."""
        if self.raiz is None:
            return [], 0
        encontradas, pendientes, calculadas = [], [self.raiz], 0
        while pendientes:
            p, hijos = pendientes.pop()
            d = distancia(palabra, p)
            calculadas += 1
            if d <= tolerancia:
                encontradas.append((d, p))
            pendientes.extend(h for k, h in hijos.items() if d - tolerancia <= k <= d + tolerancia)
        return sorted(encontradas), calculadas


def tolerancia(palabra):
    """Errores permitidos según el largo: ninguno hasta 4 letras, 1 hasta 7, 2 desde 8."""
    return 0 if len(palabra) <= 4 else 1 if len(palabra) <= 7 else 2


class Corrector:
    def __init__(self, frecuencias, zipf=None):
        """frecuencias: {palabra normalizada: cuántas veces aparece en el vocabulario}.
        zipf: frecuencia de una palabra en el español (wordfreq) o None para usar solo el vocabulario."""
        self.frecuencias = dict(frecuencias)
        self.zipf = zipf
        self.arbol = BKTree(sorted(self.frecuencias, key=lambda p: (-self.frecuencias[p], p)))

    def agregar(self, palabras, peso=1000):
        for p in palabras:
            if p not in self.frecuencias:
                self.arbol.agregar(p)
            self.frecuencias[p] = self.frecuencias.get(p, 0) + peso

    def corregir_palabra(self, palabra):
        """La palabra corregida (o la misma si es conocida, corta o no tiene una parecida)."""
        if palabra in self.frecuencias or not palabra.isalpha():
            return palabra
        if self.zipf and self.zipf(palabra) >= ZIPF_CONOCIDA:
            return palabra
        k = tolerancia(palabra)
        if k == 0:
            return palabra
        candidatas, _ = self.arbol.buscar(palabra, k)
        if not candidatas:
            return palabra
        return max(candidatas, key=lambda c: self.puntaje(*c))[1]

    def puntaje(self, d, candidata):
        """log10 P(candidata) - costo de los errores (canal ruidoso). Sin wordfreq, la frecuencia en el vocabulario."""
        previa = self.zipf(candidata) if self.zipf else 0.0
        if not previa:
            previa = 1 + math.log10(self.frecuencias.get(candidata, 1))   # palabras técnicas que wordfreq no tiene
        return previa - COSTO_POR_ERROR * d

    def corregir(self, texto):
        """Corrige cada palabra del texto (normalizado). Devuelve (texto corregido, [(antes, después)])."""
        palabras = normalizar(texto).replace(",", " ").replace(".", " ").split()
        corregidas = [self.corregir_palabra(p) for p in palabras]
        cambios = [(a, b) for a, b in zip(palabras, corregidas) if a != b]
        return " ".join(corregidas), cambios


@lru_cache(maxsize=1)
def corrector():
    frecuencias = {}
    if RUTA_VOCABULARIO.exists():
        frecuencias = json.loads(RUTA_VOCABULARIO.read_text(encoding="utf-8"))
    return Corrector(frecuencias, _zipf())
