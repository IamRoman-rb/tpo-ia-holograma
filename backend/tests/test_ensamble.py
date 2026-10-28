import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import ensamble
import generate3d
from ensamble import EnsambleInvalido, a_modelo, generar_glb, pedir_ensamble, validar
from geometria import a_glb

EJEMPLO = json.loads((Path(__file__).parent / "datos" / "ensamble_bomba.json").read_text(encoding="utf-8"))


def nodos_del_glb(datos: bytes):
    largo = struct.unpack_from("<I", datos, 12)[0]
    return json.loads(datos[20:20 + largo])["nodes"]


class ClienteFalso:
    """Imita client.beta.messages.create del SDK de Anthropic."""

    def __init__(self, texto=None, stop_reason="end_turn"):
        self.kwargs = None
        respuesta = SimpleNamespace(stop_reason=stop_reason,
                                    content=[SimpleNamespace(type="text", text=texto)] if texto else [])

        def create(**kwargs):
            self.kwargs = kwargs
            return respuesta

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_ejemplo_valido_y_completo():
    e = validar(EJEMPLO)
    assert len(e.piezas) == 8
    assert [p.id for p in e.piezas if p.orden_desmontaje is None] == ["voluta"]   # una pieza fija


def test_glb_con_una_pieza_por_nodo():
    datos = a_glb(a_modelo(validar(EJEMPLO)), "bomba")
    assert generate3d.es_glb_valido(datos)
    nodos = nodos_del_glb(datos)
    assert len(nodos) == 8
    impulsor = next(n for n in nodos if n["name"] == "impulsor")
    assert impulsor["extras"]["nombre"] == "Impulsor"
    assert impulsor["extras"]["orden"] == 4 and impulsor["extras"]["direccion"] == [0, 0, 1]


def test_formas_invalidas_se_descartan():
    datos = copy.deepcopy(EJEMPLO)
    voluta = next(p for p in datos["piezas"] if p["id"] == "voluta")
    voluta["formas"][0]["medidas"]["radio_interior"] = 15        # tubo con agujero más grande que el tubo
    datos["piezas"][0]["formas"] = [{**datos["piezas"][0]["formas"][0], "posicion": [1, 2]}]  # vector mal formado
    with pytest.raises(EnsambleInvalido):
        validar(datos)                                             # el vector inválido rompe el esquema
    datos["piezas"].pop(0)
    e = validar(datos)
    assert len(next(p for p in e.piezas if p.id == "voluta").formas) == 2   # se descartó solo el tubo inválido


def test_pieza_sin_formas_validas_desaparece():
    datos = copy.deepcopy(EJEMPLO)
    datos["piezas"][2]["formas"][0]["medidas"]["radio"] = 9999     # tuerca fuera de rango: queda sin formas
    assert "tuerca_impulsor" not in [p.id for p in validar(datos).piezas]


def test_pedido_inapropiado_sin_piezas():
    with pytest.raises(EnsambleInvalido, match="armas"):
        validar({"nombre": "-", "descripcion": "No se generan armas.", "piezas": []})


def test_color_y_direccion_se_normalizan():
    datos = copy.deepcopy(EJEMPLO)
    datos["piezas"][5]["direccion_desmontaje"] = [0, -5, 0]
    datos["piezas"][5]["color"] = "rojo"                           # no es #rrggbb: se ignora
    base = next(p for p in validar(datos).piezas if p.id == "base")
    assert base.direccion_desmontaje == [0, -1, 0] and base.color is None


def test_pedido_al_llm_usa_salida_estructurada():
    cliente = ClienteFalso(json.dumps(EJEMPLO))
    datos, e = generar_glb("una bomba de agua", cliente)
    assert len(nodos_del_glb(datos)) == 8 and e.nombre == "Bomba centrífuga"
    assert cliente.kwargs["output_config"]["format"]["schema"] == ensamble.ESQUEMA_ENSAMBLE
    assert cliente.kwargs["messages"] == [{"role": "user", "content": "una bomba de agua"}]


def test_rechazo_del_llm():
    with pytest.raises(EnsambleInvalido):
        pedir_ensamble("algo", ClienteFalso(stop_reason="refusal"))


def test_generacion_por_piezas_queda_en_el_catalogo(monkeypatch, tmp_path):
    catalogo = tmp_path / "catalogo.json"
    catalogo.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(generate3d, "RUTA_CATALOGO", catalogo)
    monkeypatch.setattr(generate3d, "DIR_GENERADOS", tmp_path / "generados")
    monkeypatch.setenv("GEN3D_MODO", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(ensamble, "pedir_ensamble", lambda descripcion, cliente=None: EJEMPLO)

    assert generate3d.modo() == "ensamble"
    r = generate3d.generar("bomba de agua")
    item = json.loads(catalogo.read_text(encoding="utf-8"))[0]
    assert r["piezas"] == 8 and item["metodo"] == "ensamble" and item["nombre"] == "Bomba centrífuga"
    assert "bomba centrífuga" in item["sinonimos"]


def test_modo_segun_configuracion(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GEN3D_URL", raising=False)
    monkeypatch.setenv("GEN3D_MODO", "auto")
    assert generate3d.modo() == "desactivado"
    monkeypatch.setenv("GEN3D_URL", "http://colab.test")
    assert generate3d.modo() == "remoto"
    monkeypatch.setenv("GEN3D_MODO", "ensamble")
    assert generate3d.modo() == "desactivado"                      # sin clave del LLM no hay ensamble
