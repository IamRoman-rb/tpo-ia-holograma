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
    ("mostrame solo el volante", "aislar", "volante", None, None),
    ("aislá el impulsor", "aislar", "impulsor", None, None),
    ("analizala", "aislar", None, None, None),
    ("mostrame el vástago por separado", "aislar", "vastago", None, None),
    ("volvé al conjunto", "conjunto", None, None, None),
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


def test_llm_completa_la_traza():
    cliente = ClienteFalso(json.dumps({"accion": "reiniciar", "objeto": None,
                                       "parametros": {"vista": None, "factor_escala": None}, "confianza": 0.9}))
    traza = {}
    interpretar_llm("reiniciá", cliente, traza=traza)
    assert traza["fin"] == "end_turn" and traza["ms_llm"] >= 0 and "modelo" in traza


def _sin_modelo_entrenado():
    raise intent.intent_modelo.ModeloNoDisponible("prueba")


def test_sin_llm_la_traza_dice_por_que(monkeypatch):
    monkeypatch.setattr(intent, "_cliente_llm", lambda: ClienteFalso("esto no es json"))
    monkeypatch.setattr(intent.intent_modelo, "cargar", _sin_modelo_entrenado)
    r = intent.interpretar("mostrame un engranaje")
    assert r["motor"] == "reglas" and r["traza"]["error"].startswith("RespuestaInvalida")


def test_llm_rechazo_devuelve_desconocido():
    r = interpretar_llm("algo", ClienteFalso(stop_reason="refusal"))
    assert r["accion"] == "desconocido" and r["confianza"] == 0.0


def test_sin_llm_ni_modelo_entrenado_caen_las_reglas(monkeypatch):
    monkeypatch.setattr(intent, "_cliente_llm", lambda: ClienteFalso("esto no es json"))
    monkeypatch.setattr(intent.intent_modelo, "cargar", _sin_modelo_entrenado)
    r = intent.interpretar("mostrame un engranaje")
    assert r["motor"] == "reglas" and r["accion"] == "mostrar"


def test_sin_llm_se_usa_el_modelo_entrenado(monkeypatch):
    monkeypatch.setattr(intent, "_cliente_llm", lambda: ClienteFalso("esto no es json"))
    respuesta = {"accion": "rotar", "objeto": None, "parametros": {"vista": None, "factor_escala": None},
                 "confianza": 0.8}
    monkeypatch.setattr(intent.intent_modelo, "interpretar_modelo", lambda f, traza=None: dict(respuesta))
    r = intent.interpretar("giralo")
    assert r["motor"] == "modelo" and r["accion"] == "rotar" and "error" in r["traza"]


@pytest.mark.parametrize("frase, factor", [
    ("ampliá tres veces", 3.0), ("agrandalo 2 veces", 2.0), ("hacelo tres veces más chico", 0.333),
    ("hacelo más chiquito", 0.67),
])
def test_factor_con_numeros_en_palabras(frase, factor):
    assert intent._factor(intent.normalizar(frase)) == factor
