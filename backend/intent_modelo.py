"""Intérprete de pedidos entrenado (motor "modelo"): el respaldo del LLM cuando no hay clave o no hay internet.

- Acción: embeddings de la frase (el mismo modelo de oraciones que usa la búsqueda del catálogo) y una red
  entrenada con frases rioplatenses (ml/intencion/entrenar_intencion.py). La confianza es la probabilidad
  que da la red: si es baja, el visor pide confirmación (sección 16.3), algo que las reglas no pueden hacer.
- Objeto o pieza: un etiquetador entrenado marca qué palabras de la frase lo nombran. Mira el vector de cada
  palabra que da el mismo modelo de embeddings, que ya tiene en cuenta el resto de la frase: así reconoce
  objetos que no vio al entrenar ("mostrame un trombón") por cómo está armado el pedido.
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
ORDEN_ACCIONES = sorted(ACCIONES_CON_OBJETO)   # columnas de la acción en los rasgos del etiquetador
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
_PALABRA_EN_TEXTO = re.compile(_PALABRA.pattern, re.IGNORECASE)   # las mismas palabras, con su posición


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


def codificar(codificador, prefijo, frases, batch_size=64):
    """Una sola pasada por el modelo de embeddings. Para cada frase devuelve su vector (normalizado, el de la
    acción) y una matriz con un vector por palabra: el promedio de los vectores de sus sub-palabras, que el
    modelo ya calculó mirando toda la frase."""
    textos = [prefijo + f for f in frases]
    salidas = codificador.encode(textos, output_value=None, batch_size=batch_size, show_progress_bar=False)
    resultado = []
    for frase, texto, s in zip(frases, textos, salidas):
        oracion = s["sentence_embedding"].float().cpu().numpy()
        n = int(s["attention_mask"].sum())
        tokens = s["token_embeddings"][:n].float().cpu().numpy()
        desde = codificador.tokenizer(texto, return_offsets_mapping=True, truncation=True,
                                      max_length=codificador.max_seq_length)["offset_mapping"]
        vectores = []
        for m in _PALABRA_EN_TEXTO.finditer(frase):
            a, b = m.start() + len(prefijo), m.end() + len(prefijo)
            k = [j for j, (x, y) in enumerate(desde[:n]) if y > x and x < b and y > a]
            v = tokens[k].mean(axis=0) if k else np.zeros(tokens.shape[1], np.float32)
            vectores.append(v / max(float(np.linalg.norm(v)), 1e-6))
        resultado.append((oracion / max(float(np.linalg.norm(oracion)), 1e-6),
                          np.asarray(vectores, np.float32).reshape(-1, tokens.shape[1])))
    return resultado


def rasgos_contexto(vectores, accion):
    """Una fila por palabra: su vector, el de la anterior, el de la siguiente y la acción (para el etiquetador)."""
    n, dim = vectores.shape
    cero = np.zeros((1, dim), np.float32)
    anterior = np.vstack([cero, vectores[:-1]]) if n else vectores
    siguiente = np.vstack([vectores[1:], cero]) if n else vectores
    acc = np.tile(np.array([accion == a for a in ORDEN_ACCIONES], np.float32), (n, 1))
    return np.hstack([vectores, anterior, siguiente, acc])


def red(capas, x, salida="softmax"):
    """Capas densas con ReLU en las ocultas; la salida, softmax (acción) o sigmoide (etiquetador)."""
    for k, (W, b) in enumerate(capas):
        x = x @ W + b
        if k < len(capas) - 1:
            x = np.maximum(x, 0)
    if salida == "sigmoide":
        return 1 / (1 + np.exp(-x))
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- modelo

class InterpreteEntrenado:
    def __init__(self, datos, codificador):
        self.datos = datos
        self.codificador = codificador
        self.acciones = datos["acciones"]
        self.capas = [(desempaquetar(c["W"]), desempaquetar(c["b"])) for c in datos["capas"]]
        etiquetador = datos["etiquetador"]
        # "contexto": red sobre los vectores de cada palabra · "rasgos": regresión logística sobre las palabras
        # · "combinado": el promedio de los dos (el de rasgos es más preciso con las formas de pedir que vio;
        # el de contexto, con objetos y frases nuevas)
        self.tipo_etiquetador = etiquetador.get("tipo", "rasgos")
        self.usa_vectores = self.tipo_etiquetador in ("contexto", "combinado")
        if self.usa_vectores:
            self.capas_etiquetador = [(desempaquetar(c["W"]), desempaquetar(c["b"])) for c in etiquetador["capas"]]
        if self.tipo_etiquetador in ("rasgos", "combinado"):
            r = etiquetador.get("rasgos", etiquetador)
            self.pesos, self.sesgo = r["pesos"], r["sesgo"]
        self.prefijo = prefijo_consulta(datos["embeddings"])

    def analizar(self, frase):
        """Probabilidad de cada acción y, si el etiquetador los usa, los vectores de las palabras (una pasada)."""
        if self.usa_vectores:
            oracion, vectores = codificar(self.codificador, self.prefijo, [frase])[0]
        else:
            oracion = np.asarray(self.codificador.encode([self.prefijo + frase], normalize_embeddings=True)[0],
                                 dtype=np.float32)
            vectores = None
        return red(self.capas, oracion), vectores

    def probabilidades(self, frase):
        return self.analizar(frase)[0]

    def probabilidades_palabras(self, frase, accion, vectores=None):
        """Para cada palabra, la probabilidad de que sea parte del objeto."""
        ps = palabras(frase)
        if not ps:
            return []
        partes = []
        if self.usa_vectores:
            if vectores is None:
                vectores = codificar(self.codificador, self.prefijo, [frase])[0][1]
            partes.append(red(self.capas_etiquetador, rasgos_contexto(vectores, accion), "sigmoide")[:, 0])
        if self.tipo_etiquetador != "contexto":
            partes.append(np.array([1 / (1 + math.exp(-(self.sesgo + sum(self.pesos.get(k, 0.0) * v
                                                                       for k, v in rasgos(ps, i, accion).items()))))
                                    for i in range(len(ps))]))
        return np.mean(partes, axis=0).tolist()

    def objeto(self, frase, accion, vectores=None):
        """El tramo de palabras seguidas que el etiquetador marca como objeto (el de mayor probabilidad total)."""
        originales = palabras_originales(frase)
        prob = self.probabilidades_palabras(frase, accion, vectores)
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
    m = modelo or cargar()
    probs, vectores = m.analizar(frase)
    k = int(np.argmax(probs))
    accion, confianza = m.acciones[k], round(float(probs[k]), 3)
    if traza is not None:
        orden = np.argsort(probs)[::-1][:3]
        traza["probabilidades"] = {m.acciones[i]: round(float(probs[i]), 3) for i in orden}
        traza["distribucion"] = {a: float(v) for a, v in zip(m.acciones, probs)}
    objeto = m.objeto(frase, accion, vectores) if accion in ACCIONES_CON_OBJETO else None
    if traza is not None and accion in ACCIONES_CON_OBJETO:
        traza["palabras_objeto"] = [{"palabra": p, "p": round(float(v), 3)} for p, v in
                                    zip(palabras_originales(frase), m.probabilidades_palabras(frase, accion, vectores))]
    return completar(frase, accion, objeto, confianza)


def completar(frase, accion, objeto, confianza):
    """Arma la intención a partir de la acción y el objeto que dio un modelo: la vista y el factor de escala
    los sacan las reglas. La usan el intérprete entrenado y el LLM propio (llm_propio.py)."""
    import intent  # aquí y no arriba: intent.py importa este módulo
    if accion == "desconocido":
        return {**intent.DESCONOCIDO, "confianza": confianza}

    t = intent.normalizar(frase)
    if accion not in ACCIONES_CON_OBJETO:
        objeto = None
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
            **intent.parametros_de_comando(t, accion),
        },
        "confianza": confianza,
    }
