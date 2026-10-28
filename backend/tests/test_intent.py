import json
from types import SimpleNamespace

import pytest

import intent
from intent import interpretar_llm, interpretar_reglas


@pytest.mark.parametrize("frase, accion, objeto, vista, factor", [
    ("mostrame una válvula de acero", "mostrar", "valvula de acero", None, None),
    ("mostrame el motor desde atrás y achicalo un poco", "mostrar", "motor", "trasera", 0.67),
    ("quiero ver un rulemán de costado", "mostrar", "ruleman", "lateral", None),
    ("hacelo el doble de grande", "escalar", None, None, 2.0),
    ("agrandalo 3 veces", "escalar", None, None, 3.0),
    ("mostrame la parte de atrás", "vista", None, "trasera", None),
    ("vista desde arriba", "vista", None, "superior", None),
    ("girálo", "rotar", None, None, None),
    ("dale una vuelta", "rotar", None, None, None),
    ("una pieza giratoria con esferas de metal", "mostrar", "pieza giratoria con esferas de metal", None, None),
    ("volvé al inicio", "reiniciar", None, None, None),
    ("qué hora es", "desconocido", None, None, None),
    ("desarmalo", "desmontar", None, None, None),
    ("sacale una pieza", "desmontar", None, None, None),
    ("mostrame el despiece", "explotar", None, None, None),
    ("mostrame la válvula desarmada", "explotar", "valvula", None, None),
    ("armalo de nuevo", "armar", None, None, None),
])
def test_reglas(frase, accion, objeto, vista, factor):
    r = interpretar_reglas(frase)
    assert r["accion"] == accion
    assert r["objeto"] == objeto
    assert r["parametros"]["vista"] == vista
    assert r["parametros"]["factor_escala"] == factor


class ClienteFalso:
    """Imita client.beta.messages.create y guarda los argumentos recibidos."""

    def __init__(self, texto=None, stop_reason="end_turn"):
        self.kwargs = None
        contenido = [SimpleNamespace(type="text", text=texto)] if texto is not None else []
        respuesta = SimpleNamespace(stop_reason=stop_reason, content=contenido)

        def create(**kwargs):
            self.kwargs = kwargs
            return respuesta

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_llm_pide_salida_estructurada_y_parsea():
    esperado = {"accion": "escalar", "objeto": None,
                "parametros": {"vista": None, "factor_escala": 2.0}, "confianza": 0.9}
    cliente = ClienteFalso(json.dumps(esperado))
    assert interpretar_llm("hacelo el doble de grande", cliente) == esperado
    assert cliente.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert cliente.kwargs["system"] == intent.PROMPT_SISTEMA
    assert cliente.kwargs["messages"] == [{"role": "user", "content": "hacelo el doble de grande"}]


def test_llm_rechazo_devuelve_desconocido():
    r = interpretar_llm("algo", ClienteFalso(stop_reason="refusal"))
    assert r["accion"] == "desconocido" and r["confianza"] == 0.0


def test_llm_json_invalido_cae_a_reglas(monkeypatch):
    monkeypatch.setattr(intent, "_cliente_llm", lambda: ClienteFalso("esto no es json"))
    r = intent.interpretar("mostrame un engranaje")
    assert r["motor"] == "reglas" and r["accion"] == "mostrar"
