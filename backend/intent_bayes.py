"""Intérprete estadístico de pedidos (motor "bayes"): Naive Bayes multinomial sobre palabras y pares de palabras.

Por la regla de Bayes, P(acción | frase) ∝ P(acción) · Π P(palabra | acción), suponiendo (ingenuamente) que las
palabras son independientes dada la acción. Se entrena contando (ml/intencion/entrenar_bayes.py) y se calcula
con logaritmos en numpy, en microsegundos. Es el modelo estadístico clásico de clasificación de texto: sirve de
línea base y como voto en el consenso. Naive Bayes es famoso por dar probabilidades demasiado extremas (las
palabras de una frase no son independientes): por eso se calibra con una temperatura ajustada en validación.

El objeto lo sacan las reglas de intent.py (Naive Bayes no marca palabras); la vista y la escala, también.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from intent_modelo import desempaquetar, palabras

RUTA_MODELO = Path(__file__).parent / "modelos" / "bayes.json"


class ModeloNoDisponible(Exception):
    pass


def rasgos(frase):
    """Palabras sin tildes y pares de palabras seguidas (con marcas de principio y fin de la frase)."""
    ps = ["<i>", *palabras(frase), "<f>"]
    return ps[1:-1] + [f"{a}_{b}" for a, b in zip(ps, ps[1:])]


class NaiveBayes:
    def __init__(self, datos):
        self.acciones = datos["acciones"]
        self.indice = {r: i for i, r in enumerate(datos["rasgos"])}
        self.log_previa = np.asarray(datos["log_previa"], np.float32)
        self.log_verosimilitud = desempaquetar(datos["log_verosimilitud"])   # (rasgos, acciones)
        self.temperatura = datos.get("temperatura", 1.0)

    def log_posterior(self, frase):
        """log P(acción) + Σ log P(rasgo | acción), sin normalizar. Los rasgos que no vio al entrenar no cuentan."""
        filas = [self.indice[r] for r in rasgos(frase) if r in self.indice]
        return self.log_previa + (self.log_verosimilitud[filas].sum(axis=0) if filas else 0)

    def probabilidades(self, frase, calibrada=True):
        z = self.log_posterior(frase) / (self.temperatura if calibrada else 1.0)
        e = np.exp(z - z.max())
        return e / e.sum()

    def rasgos_decisivos(self, frase, accion, n=3):
        """Los rasgos que más empujaron hacia esa acción frente a las demás (para explicar la decisión)."""
        k = self.acciones.index(accion)
        puntajes = []
        for r in set(rasgos(frase)):
            if r in self.indice:
                fila = self.log_verosimilitud[self.indice[r]]
                puntajes.append((float(fila[k] - np.log(np.exp(np.delete(fila, k)).mean())), r))
        return [r for _, r in sorted(puntajes, reverse=True)[:n]]


@lru_cache(maxsize=1)
def cargar(ruta=RUTA_MODELO):
    if not Path(ruta).exists():
        raise ModeloNoDisponible("falta bayes.json: correr ml/intencion/entrenar_bayes.py")
    return NaiveBayes(json.loads(Path(ruta).read_text(encoding="utf-8")))


def interpretar_bayes(frase, modelo=None, traza=None):
    """Misma salida que intent.interpretar_llm. traza (opcional) recibe las probabilidades y los rasgos decisivos."""
    import intent
    from intent_modelo import ACCIONES_CON_OBJETO, completar
    m = modelo or cargar()
    probs = m.probabilidades(frase)
    k = int(np.argmax(probs))
    accion, confianza = m.acciones[k], round(float(probs[k]), 3)
    if traza is not None:
        orden = np.argsort(probs)[::-1][:3]
        traza["probabilidades"] = {m.acciones[i]: round(float(probs[i]), 3) for i in orden}
        traza["rasgos_decisivos"] = m.rasgos_decisivos(frase, accion)
        traza["distribucion"] = {a: float(v) for a, v in zip(m.acciones, probs)}
    objeto = None
    if accion in ACCIONES_CON_OBJETO:
        reglas = intent.interpretar_reglas(frase)
        t = intent.normalizar(frase)
        # si las reglas coinciden en la acción, su objeto ya viene limpio ("la válvula desarmada" -> "válvula")
        objeto = reglas["objeto"] if reglas["accion"] == accion else \
            intent._pieza(t) if accion == "aislar" else intent._objeto(t)
    return completar(frase, accion, objeto, confianza)
