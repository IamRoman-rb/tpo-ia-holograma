from safety import LimiteFrecuencia, validar_intencion


def intencion(**kw):
    base = {"accion": "mostrar", "objeto": "valvula",
            "parametros": {"vista": None, "factor_escala": None}, "confianza": 0.9}
    return {**base, **kw}


def test_intencion_valida_pasa():
    r = validar_intencion(intencion())
    assert r["accion"] == "mostrar" and not r["requiere_confirmacion"]


def test_accion_inventada_se_rechaza():
    assert validar_intencion(intencion(accion="borrar_todo"))["accion"] == "desconocido"


def test_vista_invalida_se_rechaza():
    r = validar_intencion(intencion(parametros={"vista": "diagonal", "factor_escala": None}))
    assert r["accion"] == "desconocido"


def test_escala_se_limita():
    r = validar_intencion(intencion(accion="escalar", objeto=None, parametros={"vista": None, "factor_escala": 50}))
    assert r["parametros"]["factor_escala"] == 5.0


def test_confianza_baja_pide_confirmacion():
    assert validar_intencion(intencion(confianza=0.3))["requiere_confirmacion"]


def test_objeto_fuera_de_dominio():
    r = validar_intencion(intencion(objeto="una pistola"))
    assert r["accion"] == "desconocido" and "dominio" in r["motivo"]


def test_bomba_de_agua_si_se_permite():
    assert validar_intencion(intencion(objeto="bomba de agua"))["accion"] == "mostrar"


def test_mostrar_sin_objeto():
    assert validar_intencion(intencion(objeto=None))["accion"] == "desconocido"


def test_limite_de_frecuencia():
    lim = LimiteFrecuencia(maximo=3, ventana_s=60)
    assert [lim.permitir("a") for _ in range(4)] == [True, True, True, False]
    assert lim.permitir("b")
