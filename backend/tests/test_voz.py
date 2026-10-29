"""Voz local (voz.py): palabra de activación, transcripción con un Whisper falso y el endpoint /voz."""
import io
import math
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import voz


def wav(segundos=1.0, frecuencia=440):
    """Un tono en WAV (16 kHz, mono): basta para probar la decodificación sin un micrófono."""
    t = np.arange(int(16000 * segundos)) / 16000
    datos = (0.3 * np.sin(2 * math.pi * frecuencia * t) * 32767).astype("<i2")
    salida = io.BytesIO()
    with wave.open(salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(datos.tobytes())
    return salida.getvalue()


class WhisperFalso:
    def __init__(self, texto, avg_logprob=-0.1):
        self.texto, self.avg_logprob, self.llamadas = texto, avg_logprob, []

    def transcribe(self, muestras, **opciones):
        self.llamadas.append(opciones)
        segmento = SimpleNamespace(text=f" {self.texto} ", tokens=[1, 2, 3, 4], avg_logprob=self.avg_logprob)
        return iter([segmento] if self.texto else []), SimpleNamespace(language="es")


@pytest.mark.parametrize("texto, activado, resto", [
    ("Holograma, mostrame un engranaje", True, "mostrame un engranaje"),
    ("olograma giralo", True, "giralo"),
    ("holo grama, ¿qué modelo respondió?", True, "¿qué modelo respondió?"),
    ("mostrame un engranaje", False, "mostrame un engranaje"),
    ("hola, cómo andás", False, "hola, cómo andás"),
])
def test_palabra_de_activacion(texto, activado, resto):
    assert voz.quitar_activacion(texto) == (activado, resto)


def test_transcribir_con_haz_y_vocabulario():
    w = WhisperFalso("mostrame un rulemán")
    r = voz.transcribir(wav(), modelo=w)
    assert r["texto"] == "mostrame un rulemán" and r["idioma"] == "es"
    assert r["confianza"] == round(math.exp(-0.1), 3) and abs(r["segundos_audio"] - 1.0) < 0.01
    opciones = w.llamadas[0]
    assert opciones["beam_size"] == voz.HACES and opciones["vad_filter"] and opciones["language"] == "es"
    assert "rulemán" in opciones["initial_prompt"]


def test_sin_voz():
    r = voz.transcribir(wav(0.5), modelo=WhisperFalso(""))
    assert r["sin_voz"] and r["confianza"] == 0.0


@pytest.fixture
def cliente(monkeypatch):
    falso = WhisperFalso("holograma, giralo")
    monkeypatch.setattr(voz, "disponible", lambda: True)
    monkeypatch.setattr(voz, "cargar", lambda modelo=voz.MODELO: falso)
    main.limite._pedidos.clear()
    return TestClient(main.app)


def test_endpoint_voz(cliente):
    r = cliente.post("/voz?activacion=true", content=wav(), headers={"Content-Type": "audio/wav"}).json()
    assert r["activado"] and r["texto"] == "giralo"
    sin_activacion = cliente.post("/voz", content=wav(), headers={"Content-Type": "audio/wav"}).json()
    assert sin_activacion["texto"] == "holograma, giralo" and "activado" not in sin_activacion


def test_endpoint_voz_rechaza(cliente):
    assert cliente.post("/voz", content=b"").status_code == 422
    assert cliente.post("/voz", content=b"esto no es audio").status_code == 422
