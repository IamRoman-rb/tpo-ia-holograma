"""Intérprete de pedidos entrenado (motor "modelo"): el respaldo del LLM cuando no hay clave o no hay internet.

- Acción: embeddings de la frase (el mismo modelo de oraciones que usa la búsqueda del catálogo) y una red
  entrenada con frases rioplatenses (ml/intencion/entrenar_intencion.py). La confianza es la probabilidad
  que da la red: si es baja, el visor pide confirmación (sección 16.3), algo que las reglas no pueden hacer.
- Objeto o pieza: un etiquetador entrenado marca qué palabras de la frase lo nombran.
- Vista y factor de escala: conjuntos cerrados, los resuelven las reglas de intent.py.

Los pesos están en backend/modelos/intencion.json (JSON, sin pickle). La cuenta se hace con numpy.
"""
import base64
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

RUTA_MODELO = Path(__file__).parent / "modelos" / "intencion.json"
ACCIONES_CON_OBJETO = {"mostrar", "explotar", "aislar"}
UMBRAL_PALABRA = 0.5   # probabilidad mínima para que una palabra cuente como parte del objeto


class ModeloNoDisponible(Exception):
    pass


# Los pesos se guardan como float16 en base64: el JSON pesa 4 veces menos y la precisión alcanza de sobra
# para clasificar (la diferencia con float32 se controla en backend/tests/test_intent_modelo.py).
def empaquetar(matriz):
    m = np.asarray(matriz, dtype=np.float16)
    return {"forma": list(m.shape), "float16": base64.b64encode(m.tobytes()).decode("ascii")}


def desempaquetar(d):
    if isinstance(d, list):
        return np.asarray(d, dtype=np.float32)
    datos = np.frombuffer(base64.b64decode(d["float16"]), dtype=np.float16)
    return datos.reshape(d["forma"]).astype(np.float32)


def prefijo_consulta(nombre_embeddings):
    """Los modelos E5 se entrenaron con este prefijo para las consultas (ver retrieval._prefijos)."""
    return "query: " if "e5" in nombre_embeddings.lower() else ""


# ---------------------------------------------------------------- palabras y rasgos del etiquetador

_PALABRA = re.compile(r"[0-9a-záéíóúüñ]+")


def _sin_tildes(p):
    return "".join(c for c in unicodedata.normalize("NFD", p) if unicodedata.category(c) != "Mn")


def palabras_originales(frase):
    """Palabras de la frase en minúsculas, con tildes (para devolver el objeto tal como se escribió)."""
    return _PALABRA.findall(frase.lower())


def palabras(frase):
    """Las mismas palabras, sin tildes: con ellas se calculan los rasgos."""
    return [_sin_tildes(p) for p in palabras_originales(frase)]


def rasgos(ps, i, accion):
    """Rasgos de la palabra i para decidir si es parte del objeto: ella, sus vecinas, su forma y la acción."""
    p = ps[i]
    r = {f"p={p}": 1, f"suf3={p[-3:]}": 1, f"acc={accion}": 1, f"pos={min(i, 6)}": 1,
         "num": float(p.isdigit()), "primera": float(i == 0), "ultima": float(i == len(ps) - 1)}
    for d in (-2, -1, 1, 2):
        j = i + d
        r[f"p{d:+d}={ps[j] if 0 <= j < len(ps) else '<borde>'}"] = 1
    r[f"p-1|p={ps[i - 1] if i else '<borde>'}|{p}"] = 1
    r[f"acc|p={accion}|{p}"] = 1
    return r


# ---------------------------------------------------------------- modelo

class InterpreteEntrenado:
    def __init__(self, datos, codificador):
        self.datos = datos
        self.codificador = codificador
        self.acciones = datos["acciones"]
        self.capas = [(desempaquetar(c["W"]), desempaquetar(c["b"])) for c in datos["capas"]]
        self.pesos = datos["etiquetador"]["pesos"]
        self.sesgo = datos["etiquetador"]["sesgo"]
        self.prefijo = prefijo_consulta(datos["embeddings"])

    def probabilidades(self, frase):
        a = np.asarray(self.codificador.encode([self.prefijo + frase], normalize_embeddings=True)[0],
                       dtype=np.float32)
        for k, (W, b) in enumerate(self.capas):
            a = a @ W + b
            if k < len(self.capas) - 1:
                a = np.maximum(a, 0)
        e = np.exp(a - a.max())
        return e / e.sum()

    def objeto(self, frase, accion):
        """El tramo de palabras seguidas que el etiquetador marca como objeto (el de mayor probabilidad total)."""
        ps, originales = palabras(frase), palabras_originales(frase)
        prob = []
        for i in range(len(ps)):
            z = self.sesgo + sum(self.pesos.get(k, 0.0) * v for k, v in rasgos(ps, i, accion).items())
            prob.append(1 / (1 + math.exp(-z)))
        mejor, actual = (0.0, None), None
        for i, p in enumerate(prob + [0.0]):
            if p >= UMBRAL_PALABRA:
                actual = (actual[0], i, actual[2] + p) if actual else (i, i, p)
            elif actual:
                if actual[2] > mejor[0]:
                    mejor = (actual[2], actual)
                actual = None
        if not mejor[1]:
            return None
        inicio, fin, _ = mejor[1]
        return " ".join(originales[inicio:fin + 1])


@lru_cache(maxsize=1)
def cargar(ruta=RUTA_MODELO):
    if not Path(ruta).exists():
        raise ModeloNoDisponible(f"falta {Path(ruta).name}: correr ml/intencion/entrenar_intencion.py")
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    try:
        from retrieval import cargar_embeddings
        codificador = cargar_embeddings(datos["embeddings"])
    except Exception as e:  # sentence-transformers no instalado o el modelo no está descargado
        raise ModeloNoDisponible(f"embeddings no disponibles: {e}") from e
    return InterpreteEntrenado(datos, codificador)


def interpretar_modelo(frase, modelo=None, traza=None):
    """Misma salida que intent.interpretar_llm. traza (opcional) recibe la probabilidad de cada acción."""
    import intent  # aquí y no arriba: intent.py importa este módulo
    m = modelo or cargar()
    probs = m.probabilidades(frase)
    k = int(np.argmax(probs))
    accion, confianza = m.acciones[k], round(float(probs[k]), 3)
    if traza is not None:
        orden = np.argsort(probs)[::-1][:3]
        traza["probabilidades"] = {m.acciones[i]: round(float(probs[i]), 3) for i in orden}
    if accion == "desconocido":
        return {**intent.DESCONOCIDO, "confianza": confianza}

    t = intent.normalizar(frase)
    objeto = m.objeto(frase, accion) if accion in ACCIONES_CON_OBJETO else None
    if accion == "mostrar" and not objeto:
        objeto = intent._objeto(t)   # respaldo: lo que esté después del verbo
    # La vista y la escala se buscan fuera del nombre del objeto ("caja reductora" no es "reducí")
    if objeto:
        t = t.replace(intent.normalizar(objeto), " ")
    return {
        "accion": accion,
        "objeto": objeto,
        "parametros": {
            "vista": intent._vista(t) if accion in ("mostrar", "vista", "explotar") else None,
            "factor_escala": intent._factor(t) if accion in ("escalar", "mostrar") else None,
        },
        "confianza": confianza,
    }
