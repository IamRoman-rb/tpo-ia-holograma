"""Intérprete de pedidos con deep learning (motor "transformer"): un codificador transformer ajustado entero.

A diferencia del intérprete con embeddings (intent_modelo.py), que usa el modelo de oraciones congelado y le
pone encima clasificadores chicos, acá se ajustan TODOS los pesos de un transformer preentrenado (cientos de
millones de parámetros) con las frases de entrenamiento: el modelo aprende las dos tareas a la vez.
  - Acción: una capa lineal sobre el primer token (<s> o [CLS]), que el transformer usa para resumir la frase.
  - Objeto: etiquetado de secuencias BIO por palabra (B = empieza el objeto, I = sigue, O = no es el objeto),
    con una capa lineal sobre el vector de la primera sub-palabra de cada palabra.
  - Decodificación con Viterbi: búsqueda exacta, por programación dinámica, de la secuencia BIO de mayor
    probabilidad que sea válida (una I no puede venir después de una O). Es mejor que decidir palabra por
    palabra, que puede dejar objetos cortados ("silla de [O] oficina").
La vista y la escala, como en los otros motores, las sacan las reglas (intent_modelo.completar).

Se entrena con ml/intencion_transformer/entrenar_transformer.py; los pesos van a backend/modelos/transformer/
(safetensors, sin pickle; no se versionan por el tamaño).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from intent_modelo import _PALABRA_EN_TEXTO, palabras_originales

RUTA_MODELO = Path(__file__).parent / "modelos" / "transformer"
ETIQUETAS = ["O", "B", "I"]
# Transiciones permitidas entre etiquetas BIO (de -> a): una I solo puede seguir a una B o a otra I
PERMITIDAS = np.array([[1, 1, 0],    # desde O
                       [1, 1, 1],    # desde B
                       [1, 1, 1]],   # desde I
                      dtype=bool)
INICIO = np.array([1, 1, 0], dtype=bool)   # la primera palabra no puede ser I


class ModeloNoDisponible(Exception):
    pass


def viterbi(log_prob):
    """log_prob: (palabras, 3) log-probabilidades de O, B, I. Devuelve la secuencia válida más probable."""
    n = len(log_prob)
    if n == 0:
        return []
    menos_inf = -1e9
    puntaje = np.where(INICIO, log_prob[0], menos_inf)
    atras = np.zeros((n, 3), dtype=int)
    transicion = np.where(PERMITIDAS, 0.0, menos_inf)
    for t in range(1, n):
        candidatos = puntaje[:, None] + transicion          # (desde, hacia)
        atras[t] = candidatos.argmax(axis=0)
        puntaje = candidatos.max(axis=0) + log_prob[t]
    camino = [int(puntaje.argmax())]
    for t in range(n - 1, 0, -1):
        camino.append(int(atras[t][camino[-1]]))
    return camino[::-1]


def tramos(camino):
    """Tramos (inicio, fin) de palabras marcadas B, I, I…"""
    salida, inicio = [], None
    for i, e in enumerate(camino + [0]):
        if e == 1 or (e == 0 and inicio is not None):
            if inicio is not None:
                salida.append((inicio, i - 1))
            inicio = i if e == 1 else None
    return salida


def posiciones_palabras(frase, desplazamientos):
    """Para cada palabra de la frase (las mismas que intent_modelo.palabras), el índice de su primer token."""
    primeros = []
    for m in _PALABRA_EN_TEXTO.finditer(frase):
        k = next((j for j, (a, b) in enumerate(desplazamientos) if b > a and a < m.end() and b > m.start()), None)
        primeros.append(k)
    return primeros


def crear_red(base, n_acciones):
    """El codificador preentrenado con las dos cabezas (acción y BIO). La usa también el entrenamiento."""
    import torch
    from transformers import AutoModel

    class Interprete(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codificador = AutoModel.from_pretrained(base) if isinstance(base, str) else base
            h = self.codificador.config.hidden_size
            self.caida = torch.nn.Dropout(0.1)
            self.accion = torch.nn.Linear(h, n_acciones)
            self.bio = torch.nn.Linear(h, len(ETIQUETAS))

        def forward(self, input_ids, attention_mask):
            x = self.caida(self.codificador(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
            return self.accion(x[:, 0]), self.bio(x)

    return Interprete()


class InterpreteTransformer:
    def __init__(self, ruta=RUTA_MODELO):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.torch = torch
        self.config = json.loads((Path(ruta) / "interprete.json").read_text(encoding="utf-8"))
        self.acciones = self.config["acciones"]
        self.tok = AutoTokenizer.from_pretrained(ruta)
        codificador = AutoModel.from_config(AutoConfig.from_pretrained(ruta))
        self.red = crear_red(codificador, len(self.acciones))
        self.red.load_state_dict(load_file(str(Path(ruta) / "interprete.safetensors")))
        self.red.eval()
        self.temperatura = self.config.get("temperatura", 1.0)
        self.nombre = f"{self.config['base']} ajustado ({self.config['parametros'] / 1e6:.0f} M de parámetros)"

    def analizar(self, frase):
        """Probabilidad de cada acción (calibrada) y log-probabilidades BIO de cada palabra."""
        x = self.tok(frase, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=64)
        desplazamientos = x.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            logits_accion, logits_bio = self.red(x["input_ids"], x["attention_mask"])
        z = logits_accion[0].numpy() / self.temperatura
        p = np.exp(z - z.max())
        p /= p.sum()
        bio = self.torch.log_softmax(logits_bio[0], dim=-1).numpy()
        primeros = posiciones_palabras(frase, desplazamientos)
        log_prob = np.array([bio[k] if k is not None else np.log([0.98, 0.01, 0.01]) for k in primeros])
        return p, log_prob

    def objeto(self, frase, log_prob):
        """El tramo de la mejor secuencia BIO (Viterbi). Si hay varios, el de mayor probabilidad media."""
        camino = viterbi(log_prob)
        candidatos = tramos(camino)
        if not candidatos:
            return None
        def puntaje(t):
            a, b = t
            return float(np.mean([log_prob[i][camino[i]] for i in range(a, b + 1)]))
        a, b = max(candidatos, key=puntaje)
        return " ".join(palabras_originales(frase)[a:b + 1])


@lru_cache(maxsize=1)
def cargar(ruta=RUTA_MODELO):
    if not (Path(ruta) / "interprete.json").exists():
        raise ModeloNoDisponible("falta backend/modelos/transformer: correr ml/intencion_transformer/entrenar_transformer.py")
    try:
        return InterpreteTransformer(ruta)
    except ImportError as e:
        raise ModeloNoDisponible(f"no se pudo cargar: {e}") from e


def interpretar_transformer(frase, modelo=None, traza=None):
    """Misma salida que intent.interpretar_llm. traza (opcional) recibe la distribución de la acción."""
    from intent_modelo import ACCIONES_CON_OBJETO, completar
    m = modelo or cargar()
    probs, log_prob = m.analizar(frase)
    k = int(np.argmax(probs))
    accion, confianza = m.acciones[k], round(float(probs[k]), 3)
    if traza is not None:
        orden = np.argsort(probs)[::-1][:3]
        traza["probabilidades"] = {m.acciones[i]: round(float(probs[i]), 3) for i in orden}
        traza["distribucion"] = {a: float(v) for a, v in zip(m.acciones, probs)}
        traza["transformer"] = m.nombre
    objeto = m.objeto(frase, log_prob) if accion in ACCIONES_CON_OBJETO else None
    if traza is not None and accion in ACCIONES_CON_OBJETO:
        camino = viterbi(log_prob)
        traza["bio"] = [{"palabra": p, "etiqueta": ETIQUETAS[e], "p": round(float(np.exp(log_prob[i][e])), 3)}
                        for i, (p, e) in enumerate(zip(palabras_originales(frase), camino))]
    return completar(frase, accion, objeto, confianza)

