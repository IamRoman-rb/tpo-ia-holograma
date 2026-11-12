"""Respuestas habladas con Piper (licencia MIT), en el mismo equipo y sin internet.

Antes las leía la síntesis del navegador, que depende de las voces instaladas en el sistema: en un Windows sin voces
en español leía con acento inglés, y las voces "naturales" de Chrome y Edge necesitan internet. Piper es una red
neuronal de texto a voz (VITS exportada a ONNX) que corre en el procesador en décimas de segundo.

La voz es la de un asistente propio: masculina, pausada y estable. Se logra con los parámetros de síntesis:
  - length_scale > 1: habla un poco más lento;
  - noise_scale y noise_w_scale bajos: menos variación de entonación y de ritmo, una voz más pareja y "serena".
El frontend le suma un leve tratamiento de sonido (ecualización y un eco muy corto) para que suene a asistente.

Variables: VOZ_RESPUESTA (es_MX-ald-medium por defecto; también es_ES-davefx-medium o es_AR-daniela-high),
VOZ_RESPUESTA_MODO=desactivado para no usarla (el navegador vuelve a leer con sus voces).
La primera vez descarga la voz de Hugging Face (~60 MB); después alcanza con la caché (HF_HUB_OFFLINE=1).
"""
import io
import logging
import os
import threading
import time
import wave
from collections import OrderedDict
from functools import lru_cache

log = logging.getLogger(__name__)

VOCES = {"es_MX-ald-medium": "es/es_MX/ald/medium",
         "es_ES-davefx-medium": "es/es_ES/davefx/medium",
         "es_AR-daniela-high": "es/es_AR/daniela/high"}
VOZ = os.getenv("VOZ_RESPUESTA") or "es_MX-ald-medium"
MAXIMO_CARACTERES = 300
RITMO = 1.04          # length_scale: >1 más lento (el frontend además lo baja un 3 %)
VARIACION = 0.45      # noise_scale: entonación
VARIACION_RITMO = 0.6  # noise_w_scale: duración de cada sonido

_bloqueo = threading.Lock()
_bloqueo_carga = threading.Lock()   # la precarga y el primer pedido no cargan la voz dos veces
_memoria = OrderedDict()   # las frases fijas ("¿Sí?", "No entendí el pedido.") se sintetizan una sola vez
MAXIMO_MEMORIA = 64


class VozRespuestaNoDisponible(Exception):
    pass


def activada():
    return (os.getenv("VOZ_RESPUESTA_MODO") or "").lower() != "desactivado" and VOZ in VOCES


def disponible():
    """Sin cargar la voz: ¿está instalada y no desactivada?"""
    if not activada():
        return False
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def cargar(nombre=VOZ):
    if not activada():
        raise VozRespuestaNoDisponible("VOZ_RESPUESTA_MODO=desactivado o voz desconocida")
    try:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
    except ImportError as e:
        raise VozRespuestaNoDisponible(f"falta piper-tts: {e}") from e
    t0 = time.perf_counter()
    try:
        modelo = hf_hub_download("rhasspy/piper-voices", f"{VOCES[nombre]}/{nombre}.onnx")
        hf_hub_download("rhasspy/piper-voices", f"{VOCES[nombre]}/{nombre}.onnx.json")
    except Exception as e:   # sin internet y sin la voz en la caché
        raise VozRespuestaNoDisponible(f"no se pudo obtener la voz {nombre} ({type(e).__name__})") from e
    voz = PiperVoice.load(modelo)
    log.info("Voz de respuesta %s cargada en %.1f s", nombre, time.perf_counter() - t0)
    return voz


def sintetizar(texto: str) -> bytes:
    """Texto -> WAV (bytes). Las frases repetidas salen de la memoria."""
    texto = " ".join(texto.split())[:MAXIMO_CARACTERES]
    if not texto:
        raise ValueError("texto vacío")
    if texto in _memoria:
        _memoria.move_to_end(texto)
        return _memoria[texto]
    from piper import SynthesisConfig
    with _bloqueo_carga:
        voz = cargar()
    config = SynthesisConfig(length_scale=RITMO, noise_scale=VARIACION, noise_w_scale=VARIACION_RITMO)
    salida = io.BytesIO()
    with _bloqueo, wave.open(salida, "wb") as wav:
        voz.synthesize_wav(texto, wav, syn_config=config)
    audio = salida.getvalue()
    _memoria[texto] = audio
    if len(_memoria) > MAXIMO_MEMORIA:
        _memoria.popitem(last=False)
    return audio
