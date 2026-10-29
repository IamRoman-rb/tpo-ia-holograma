"""LLM propio: Qwen2.5-Instruct (licencia Apache 2.0) ajustado por el equipo para interpretar pedidos.

Corre en el equipo, sin internet ni costo por pedido. Se entrena con ml/llm_propio/entrenar_llm.py en dos
tamaños: "grande" (1.500 millones de parámetros, backend/modelos/llm_propio_grande/, ~3 GB) y "chico" (500
millones, backend/modelos/llm_propio/, ~1 GB); ninguno se versiona. Se usa el grande si está (es más preciso),
salvo que LLM_PROPIO=chico pida el rápido. Escribe "accion|objeto" (un formato corto, porque en el
procesador cada token tarda ~70 ms); la vista y el factor de escala los sacan las reglas, como en el
intérprete entrenado. La confianza es la probabilidad que el modelo le dio a las palabras de la acción.
"""
import math
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import get_args

from schemas import Accion

RUTAS = {"grande": Path(__file__).parent / "modelos" / "llm_propio_grande",
         "chico": Path(__file__).parent / "modelos" / "llm_propio"}


def _ruta_elegida():
    pedido = (os.getenv("LLM_PROPIO") or "").lower()
    if pedido in RUTAS:
        return RUTAS[pedido]
    return next((r for r in RUTAS.values() if (r / "config.json").exists()), RUTAS["chico"])


RUTA_MODELO = _ruta_elegida()
ACCIONES = set(get_args(Accion))
MAX_TOKENS = 24   # alcanza para la acción y un objeto largo ("silla de oficina con ruedas")


class LlmNoDisponible(Exception):
    pass


class RespuestaInvalida(Exception):
    pass


def entrada(frase):
    """Lo que lee el modelo (el mismo texto al entrenar y al usarlo)."""
    return f"Pedido: {frase.strip()}\nRespuesta:"


def respuesta(accion, objeto):
    """Lo que tiene que escribir: la acción y el objeto, o "-" si el pedido no nombra ninguno."""
    return f" {accion}|{(objeto or '').strip() or '-'}"


def palabras(texto):
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


class LlmPropio:
    def __init__(self, ruta=RUTA_MODELO):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ruta)
        self.modelo = AutoModelForCausalLM.from_pretrained(ruta, dtype=torch.float32)
        self.modelo.eval()
        n = sum(p.numel() for p in self.modelo.parameters())
        self.nombre = f"Qwen2.5-{n / 1e9:.1f}B ajustado (LLM propio)"

    def distribucion(self, frase, acciones):
        """P(acción | pedido) para TODAS las acciones: se le hace "escribir" a la fuerza cada una (" mostrar|",
        " rotar|", ...) y se suma la log-probabilidad de sus tokens. Una sola pasada en lote. La usa el consenso."""
        torch = self.torch
        base = self.tok(entrada(frase)).input_ids
        continuaciones = [self.tok(f" {a}|").input_ids for a in acciones]
        secuencias = [base + c for c in continuaciones]
        largo = max(map(len, secuencias))
        relleno = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.tensor([s + [relleno] * (largo - len(s)) for s in secuencias])
        mascara = torch.tensor([[1] * len(s) + [0] * (largo - len(s)) for s in secuencias])
        with torch.no_grad():
            log_p = torch.log_softmax(self.modelo(input_ids=ids, attention_mask=mascara).logits.float(), dim=-1)
        puntajes = []
        for i, c in enumerate(continuaciones):
            # el token j de la continuación se predice en la posición anterior
            puntajes.append(sum(float(log_p[i, len(base) + j - 1, t]) for j, t in enumerate(c)))
        z = [p - max(puntajes) for p in puntajes]
        total = sum(math.exp(v) for v in z)
        return {a: math.exp(v) / total for a, v in zip(acciones, z)}

    def generar(self, frase):
        """Decodificación voraz (siempre la palabra más probable). Devuelve acción, objeto, confianza y tokens."""
        x = self.tok(entrada(frase), return_tensors="pt")
        with self.torch.no_grad():
            s = self.modelo.generate(**x, max_new_tokens=MAX_TOKENS, do_sample=False, output_scores=True,
                                     return_dict_in_generate=True, pad_token_id=self.tok.eos_token_id)
        nuevos = s.sequences[0, x.input_ids.shape[1]:]
        texto = self.tok.decode(nuevos, skip_special_tokens=True).split("\n")[0].strip()
        accion, _, objeto = texto.partition("|")
        accion, objeto = accion.strip(), objeto.strip()
        # Confianza: el producto de las probabilidades de los tokens hasta la barra (la acción)
        log_p, escrito = 0.0, ""
        for token, puntajes in zip(nuevos, s.scores):
            log_p += float(self.torch.log_softmax(puntajes[0], dim=-1)[token])
            escrito += self.tok.decode([token])
            if "|" in escrito:
                break
        return {"accion": accion, "objeto": None if objeto in ("", "-") else objeto,
                "confianza": round(math.exp(log_p), 3), "texto": texto, "tokens_entrada": x.input_ids.shape[1],
                "tokens_salida": len(nuevos)}


@lru_cache(maxsize=1)
def cargar(ruta=RUTA_MODELO):
    if not (Path(ruta) / "config.json").exists():
        raise LlmNoDisponible("falta backend/modelos/llm_propio: correr ml/llm_propio/entrenar_llm.py")
    try:
        return LlmPropio(ruta)
    except ImportError as e:  # transformers o torch no instalados
        raise LlmNoDisponible(f"no se pudo cargar: {e}") from e


def interpretar_llm_propio(frase, modelo=None, traza=None, distribucion=False):
    """Misma salida que intent.interpretar_llm. traza (opcional) recibe lo que escribió el modelo y el tiempo;
    con distribucion=True, también la probabilidad de cada acción (cuesta una pasada más: la usa el consenso)."""
    import intent_modelo
    m = modelo or cargar()
    t0 = time.perf_counter()
    r = m.generar(frase)
    if distribucion and traza is not None:
        traza["distribucion"] = m.distribucion(frase, sorted(ACCIONES))
    if traza is not None:
        traza.update(llm_propio=m.nombre, crudo_llm_propio=r["texto"], tokens_entrada=r["tokens_entrada"],
                     tokens_salida=r["tokens_salida"], ms_llm_propio=round((time.perf_counter() - t0) * 1000))
    if r["accion"] not in ACCIONES:
        raise RespuestaInvalida(f"acción desconocida: {r['texto']!r}")
    return intent_modelo.completar(frase, r["accion"], r["objeto"], r["confianza"])
