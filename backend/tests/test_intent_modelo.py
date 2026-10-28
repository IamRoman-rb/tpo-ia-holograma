"""Intérprete de pedidos entrenado (intent_modelo.py): pesos empaquetados, etiquetador del objeto y cadena completa."""
import json

import numpy as np
import pytest

import intent_modelo
from intent_modelo import InterpreteEntrenado, desempaquetar, empaquetar, interpretar_modelo


def test_empaquetar_en_float16_conserva_los_pesos():
    W = np.random.default_rng(0).normal(0, 0.3, (40, 7)).astype(np.float32)
    d = desempaquetar(json.loads(json.dumps(empaquetar(W))))
    assert d.shape == W.shape and np.max(np.abs(d - W)) < 1e-3


class CodificadorFalso:
    """Devuelve un vector que señala si la frase tiene 'gira' (para no cargar un modelo de embeddings)."""
    def encode(self, frases, normalize_embeddings=True):
        return np.array([[1.0, 0.0] if "gira" in f else [0.0, 1.0] for f in frases], dtype=np.float32)


def modelo_falso():
    datos = {"embeddings": "falso", "acciones": ["rotar", "mostrar"],
             "capas": [{"W": empaquetar(np.array([[4.0, -4.0], [-4.0, 4.0]])), "b": [0.0, 0.0]}],
             "etiquetador": {"pesos": {"p=turbina": 4.0, "p-1=la": 2.0}, "sesgo": -2.0}}
    return InterpreteEntrenado(datos, CodificadorFalso())


def test_la_accion_y_su_confianza_salen_de_la_red():
    r = interpretar_modelo("giralo", modelo_falso())
    assert r["accion"] == "rotar" and r["confianza"] > 0.99 and r["objeto"] is None


def test_el_objeto_lo_marca_el_etiquetador():
    traza = {}
    r = interpretar_modelo("mostrame la turbina desde atrás", modelo_falso(), traza)
    assert r["accion"] == "mostrar" and r["objeto"] == "turbina"
    assert r["parametros"]["vista"] == "trasera"
    assert list(traza["probabilidades"]) == ["mostrar", "rotar"]


# Con el modelo exportado de verdad (backend/modelos/intencion.json). El etiquetador no necesita embeddings.
RUTA = intent_modelo.RUTA_MODELO
con_modelo = pytest.mark.skipif(not RUTA.exists(), reason="falta backend/modelos/intencion.json")


@con_modelo
@pytest.mark.parametrize("frase, accion, objeto", [
    ("mostrame la turbina de acero desde atrás", "mostrar", "turbina de acero"),
    ("quiero ver un compresor de dos etapas", "mostrar", "compresor de dos etapas"),
    ("analizá solamente el vástago", "aislar", "vástago"),
    ("mostrame el despiece de la caja reductora", "explotar", "caja reductora"),
])
def test_etiquetador_entrenado(frase, accion, objeto):
    datos = json.loads(RUTA.read_text(encoding="utf-8"))
    assert InterpreteEntrenado(datos, codificador=None).objeto(frase, accion) == objeto


@con_modelo
def test_modelo_completo_si_hay_embeddings():
    try:
        m = intent_modelo.cargar()
    except intent_modelo.ModeloNoDisponible as e:
        pytest.skip(str(e))
    assert interpretar_modelo("dale una vueltita", m)["accion"] == "rotar"
    r = interpretar_modelo("necesito ver una polea de aluminio de costado", m)
    assert r["accion"] == "mostrar" and "polea" in r["objeto"] and r["parametros"]["vista"] == "lateral"
