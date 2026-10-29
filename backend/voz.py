"""Reconocimiento de voz local con Whisper (faster-whisper, licencia MIT): el audio no sale del equipo.

Antes la voz la reconocía el navegador, que en Chrome y Edge manda el audio a Google o Microsoft. La propuesta
(sección 16.1) pide preferir la transcripción local: acá Whisper large-v3 corre en el procesador (int8, con
CTranslate2), gratis y sin internet después de la primera descarga.

- Búsqueda en haz (beam search, 5 hipótesis): Whisper no se queda con la palabra más probable en cada paso,
  sigue las 5 mejores frases parciales y se queda con la de mayor probabilidad total.
- Detector de voz (VAD de Silero, incluido en faster-whisper): saca los silencios antes de transcribir; sin él,
  Whisper "escucha" frases en el ruido.
- Un texto inicial con el vocabulario del visor (verbos, piezas del catálogo, nombres de los modelos) inclina la
  transcripción hacia palabras del dominio ("rulemán", "despiece", "Naive Bayes").
- Palabra de activación para el modo manos libres: solo se atiende lo que empieza con "holograma" (se tolera
  que Whisper la escriba distinto: "olograma", "holo grama"; búsqueda difusa de busqueda_difusa.py).
La confianza es exp(log-probabilidad media por token) del texto elegido.

Variables: WHISPER_MODELO (large-v3 por defecto; large-v3-turbo es ~4 veces más rápido y apenas menos preciso),
VOZ_MODO=desactivado para no cargarlo.
"""
import io
import logging
import math
import os
import re
import threading
import time
from functools import lru_cache

from busqueda_difusa import distancia
from texto import normalizar

log = logging.getLogger(__name__)

MODELO = os.getenv("WHISPER_MODELO") or "large-v3"
HACES = 5
MAXIMO_SEGUNDOS = 30
PALABRA_ACTIVACION = "holograma"
TEXTO_INICIAL = ("Pedidos a un visor de hologramas, en español rioplatense. Holograma, mostrame un engranaje recto. "
                 "Giralo, agrandalo, desarmalo, mostrame el despiece del rulemán, aislá el volante, volvé al "
                 "conjunto. Usá el árbol de decisión para los gestos. Pasá a Naive Bayes, al transformer o al LLM "
                 "propio. ¿Qué modelo respondió? ¿Qué tan seguro estás?")

_bloqueo = threading.Lock()


class VozNoDisponible(Exception):
    pass


def activada():
    return (os.getenv("VOZ_MODO") or "").lower() != "desactivado"


@lru_cache(maxsize=1)
def cargar(modelo=MODELO):
    if not activada():
        raise VozNoDisponible("VOZ_MODO=desactivado")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise VozNoDisponible(f"falta faster-whisper: {e}") from e
    t0 = time.perf_counter()
    m = WhisperModel(modelo, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    log.info("Whisper %s cargado en %.0f s", modelo, time.perf_counter() - t0)
    return m


def disponible():
    """Sin cargar el modelo: ¿está instalado y no desactivado?"""
    if not activada():
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def quitar_activacion(texto):
    """¿Empieza con la palabra de activación? Devuelve (activado, resto del texto). Tolera dos letras de error
    y que venga partida en dos palabras ("holo grama")."""
    palabras = re.findall(r"\S+", texto.strip())
    for n in (1, 2):
        if len(palabras) >= n:
            candidata = normalizar("".join(palabras[:n])).replace(",", "").replace(".", "")
            if distancia(candidata, PALABRA_ACTIVACION, maximo=2) <= 2:
                resto = " ".join(palabras[n:]).lstrip(" ,.;:")
                return True, resto
    return False, texto.strip()


def transcribir(audio, modelo=None, hipotesis=HACES):
    """audio: bytes de un archivo (webm, ogg, wav...) o ruta. Devuelve texto, confianza, idioma, duración y tiempo."""
    from faster_whisper import decode_audio
    m = modelo or cargar()
    t0 = time.perf_counter()
    muestras = decode_audio(io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio, sampling_rate=16000)
    segundos = len(muestras) / 16000
    if segundos > MAXIMO_SEGUNDOS:
        muestras = muestras[: MAXIMO_SEGUNDOS * 16000]
    with _bloqueo:   # un pedido por vez: el modelo usa todos los núcleos
        segmentos, info = m.transcribe(muestras, language="es", beam_size=hipotesis, vad_filter=True,
                                       initial_prompt=TEXTO_INICIAL, condition_on_previous_text=False,
                                       temperature=0.0)
        segmentos = list(segmentos)
    texto = " ".join(s.text.strip() for s in segmentos).strip()
    tokens = sum(len(s.tokens) for s in segmentos) or 1
    log_prob = sum(s.avg_logprob * len(s.tokens) for s in segmentos) / tokens if segmentos else -math.inf
    return {"texto": texto, "confianza": round(math.exp(log_prob), 3) if segmentos else 0.0,
            "idioma": info.language, "segundos_audio": round(segundos, 2),
            "ms_transcripcion": round((time.perf_counter() - t0) * 1000), "modelo": f"whisper {MODELO}",
            "sin_voz": not texto}
