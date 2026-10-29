"""Algoritmos de búsqueda, estadísticos y de decisión: búsqueda difusa (BK-tree), Viterbi, Naive Bayes y consenso."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import busqueda_difusa as bd
import consenso
import intent_bayes
import intent_transformer as it

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import estadistica  # noqa: E402


# ---------------------------------------------------------------- búsqueda difusa

@pytest.mark.parametrize("a, b, d", [
    ("engranaje", "engranaje", 0), ("engranage", "engranaje", 1), ("rulemam", "ruleman", 1),
    ("engranjae", "engranaje", 1),        # dos letras vecinas permutadas cuentan como una (Damerau)
    ("valvula", "balbula", 2), ("", "abc", 3),
])
def test_distancia_de_edicion(a, b, d):
    assert bd.distancia(a, b) == d


def test_bk_tree_encuentra_lo_mismo_que_la_fuerza_bruta():
    palabras = ["engranaje", "engranajes", "engrase", "rodamiento", "ruleman", "valvula", "volante", "tornillo",
                "tornillos", "turbina", "tuerca", "bulon", "bomba", "brida"]
    arbol = bd.BKTree(palabras)
    for consulta in ["engranage", "tornilo", "brida", "valbula", "xyz"]:
        for k in (1, 2):
            encontradas, calculadas = arbol.buscar(consulta, k)
            esperadas = sorted((bd.distancia(consulta, p), p) for p in palabras if bd.distancia(consulta, p) <= k)
            assert encontradas == esperadas
            assert calculadas <= len(palabras)


def test_corrector_canal_ruidoso():
    c = bd.Corrector({"ruleman": 5, "engranaje": 20, "engranajes": 3, "acero": 10, "compresor": 2},
                     zipf=lambda p: {"acero": 4.0, "compresor": 3.0}.get(p, 0.0))
    assert c.corregir("mostrame un rulemam de asero")[0].endswith("ruleman de acero")
    assert c.corregir_palabra("engranage") == "engranaje"     # más cerca que "engranajes"
    assert c.corregir_palabra("compresor") == "compresor"     # palabra conocida: no se toca
    assert c.corregir_palabra("tuby") == "tuby"               # corta: no se corrige (demasiado ambiguo)


# ---------------------------------------------------------------- Viterbi (BIO)

def test_viterbi_respeta_las_transiciones():
    # Palabra por palabra ganaría O, I (inválido: una I no puede seguir a una O); Viterbi elige O, B
    lp = np.log(np.array([[0.9, 0.05, 0.05], [0.1, 0.4, 0.5]]))
    assert it.viterbi(lp) == [0, 1]
    assert it.tramos([0, 1, 2, 2, 0, 1]) == [(1, 3), (5, 5)]
    assert it.viterbi(np.zeros((0, 3))) == []


def test_viterbi_es_el_maximo_exacto():
    rng = np.random.default_rng(0)
    from itertools import product
    for _ in range(20):
        lp = np.log(rng.dirichlet(np.ones(3), size=5))
        validas = [c for c in product(range(3), repeat=5) if c[0] != 2 and
                   all(not (a == 0 and b == 2) for a, b in zip(c, c[1:]))]
        mejor = max(validas, key=lambda c: sum(lp[i, e] for i, e in enumerate(c)))
        assert it.viterbi(lp) == list(mejor)


# ---------------------------------------------------------------- Naive Bayes

con_bayes = pytest.mark.skipif(not intent_bayes.RUTA_MODELO.exists(), reason="falta bayes.json")


@con_bayes
def test_bayes_entrenado():
    traza = {}
    r = intent_bayes.interpretar_bayes("giralo un poco", traza=traza)
    assert r["accion"] == "rotar" and 0 < r["confianza"] <= 1
    assert abs(sum(traza["distribucion"].values()) - 1) < 1e-6 and traza["rasgos_decisivos"]
    assert intent_bayes.interpretar_bayes("mostrame un engranaje recto")["objeto"] == "engranaje recto"


def test_rasgos_de_bayes():
    assert intent_bayes.rasgos("Mostrame un Engranaje") == ["mostrame", "un", "engranaje", "<i>_mostrame",
                                                             "mostrame_un", "un_engranaje", "engranaje_<f>"]


# ---------------------------------------------------------------- estadística

def test_mcnemar_y_bootstrap():
    a = [1] * 30 + [0] * 10
    b = [1] * 20 + [0] * 20
    solo_a, solo_b, p = estadistica.mcnemar(a, b)
    assert (solo_a, solo_b) == (10, 0) and p < 0.01
    assert estadistica.mcnemar(a, a)[2] == 1.0
    media, desde, hasta = estadistica.intervalo_bootstrap(a)
    assert desde < media == 0.75 < hasta


def test_temperatura_y_ece():
    # Logits demasiado seguros: la temperatura óptima es mayor que 1 y baja el error de calibración
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    logits = rng.normal(size=(400, 3))
    logits[np.arange(400), y] += 1.0
    logits *= 6
    T = estadistica.ajustar_temperatura(logits, y)
    antes, despues = estadistica.con_temperatura(logits, 1), estadistica.con_temperatura(logits, T)
    acierto = antes.argmax(1) == y
    assert T > 1.5 and estadistica.ece(despues.max(1), acierto) < estadistica.ece(antes.max(1), acierto)


# ---------------------------------------------------------------- consenso

def _salida(accion, p, objeto=None):
    lista = consenso.acciones()
    dist = np.full(len(lista), (1 - p) / (len(lista) - 1))
    dist[lista.index(accion)] = p
    return {"intencion": {"accion": accion, "objeto": objeto}, "distribucion": dist, "ms": 1}


def test_combinacion_log_lineal():
    salidas = {"a": _salida("rotar", 0.6), "b": _salida("rotar", 0.6), "c": _salida("escalar", 0.99)}
    lista = consenso.acciones()
    iguales = consenso.combinar(salidas, {"pesos": {"a": 1, "b": 1, "c": 1}}, lista)
    assert lista[int(np.argmax(iguales))] == "escalar"            # c está segurísimo: le gana a dos dudosos
    sin_c = consenso.combinar(salidas, {"pesos": {"a": 1, "b": 1, "c": 0.1}}, lista)
    assert lista[int(np.argmax(sin_c))] == "rotar"               # con poco peso, c no decide
    assert math.isclose(sum(sin_c), 1)


def test_voto_del_objeto_agrupa_formas_parecidas():
    salidas = {"a": _salida("mostrar", 0.9, "silla de oficina"), "b": _salida("mostrar", 0.8, "la silla de oficina"),
               "c": _salida("mostrar", 0.95, "silla"), "d": _salida("rotar", 0.99, "otra cosa")}
    assert consenso.votar_objeto(salidas, "mostrar", {"pesos": {}}) in ("silla de oficina", "la silla de oficina")


def test_consenso_decide_si_preguntar(monkeypatch):
    monkeypatch.setattr(consenso, "pesos_guardados", lambda: None)
    seguro = {"a": _salida("rotar", 0.999), "b": _salida("rotar", 0.999)}
    dudoso = {"a": _salida("rotar", 0.5), "b": _salida("escalar", 0.45)}
    for salidas, pregunta in ((seguro, False), (dudoso, True)):
        monkeypatch.setattr(consenso, "correr_motores", lambda f, m=None, s=salidas: s)
        traza = {}
        r = consenso.interpretar_consenso("giralo", traza=traza)
        assert (r["confianza"] < r["umbral_confirmacion"]) == pregunta
        assert traza["decision"].startswith("preguntar" if pregunta else "ejecutar")
        assert len(traza["votos"]) == 2
