"""LLM propio (llm_propio.py): formato de entrada y salida, y cómo se arma la intención con lo que escribe."""
import pytest

import llm_propio
from llm_propio import RespuestaInvalida, entrada, interpretar_llm_propio, respuesta


class LlmFalso:
    """Devuelve lo que se le indique (para no cargar el modelo de 1 GB)."""
    nombre = "falso"

    def __init__(self, accion, objeto=None, confianza=0.9):
        self.r = {"accion": accion, "objeto": objeto, "confianza": confianza, "texto": f"{accion}|{objeto or '-'}",
                  "tokens_entrada": 12, "tokens_salida": 6}

    def generar(self, frase):
        return self.r


def test_formato_de_entrenamiento():
    assert entrada(" mostrame una silla ") == "Pedido: mostrame una silla\nRespuesta:"
    assert respuesta("mostrar", "silla de oficina") == " mostrar|silla de oficina"
    assert respuesta("rotar", None) == " rotar|-"


def test_la_vista_y_la_escala_las_completan_las_reglas():
    traza = {}
    r = interpretar_llm_propio("mostrame una caja reductora desde atrás", LlmFalso("mostrar", "caja reductora"), traza)
    assert r["accion"] == "mostrar" and r["objeto"] == "caja reductora" and r["confianza"] == 0.9
    # "reductora" no dispara el achicado: la escala se busca fuera del nombre del objeto
    assert r["parametros"] == {"vista": "trasera", "factor_escala": None}
    assert traza["crudo_llm_propio"] == "mostrar|caja reductora"


def test_acciones_sin_objeto_lo_descartan():
    r = interpretar_llm_propio("giralo", LlmFalso("rotar", "algo"))
    assert r["accion"] == "rotar" and r["objeto"] is None


def test_una_accion_inventada_no_pasa():
    with pytest.raises(RespuestaInvalida):
        interpretar_llm_propio("hacé algo", LlmFalso("bailar"))


con_llm = pytest.mark.skipif(not (llm_propio.RUTA_MODELO / "config.json").exists(),
                             reason="falta backend/modelos/llm_propio (ml/llm_propio/entrenar_llm.py)")


@con_llm
def test_llm_propio_entrenado():
    r = interpretar_llm_propio("che, mostrame un acordeón antiguo de costado")
    assert r["accion"] == "mostrar" and "acordeon" in llm_propio.palabras(r["objeto"] or "")
    assert interpretar_llm_propio("giralo un poco para la izquierda")["accion"] == "rotar"
