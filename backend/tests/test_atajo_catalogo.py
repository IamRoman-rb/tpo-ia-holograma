"""Atajo por nombre: un pedido que es solo el nombre de algo del catálogo se muestra sin consultar a los motores."""
import json

import pytest

import atajo_catalogo
import intent


@pytest.fixture
def catalogo(tmp_path):
    ruta = tmp_path / "catalogo.json"
    ruta.write_text(json.dumps([
        {"id": "freno_disco", "nombre": "Freno a disco", "sinonimos": ["disco de freno", "caliper"]},
        {"id": "turbocompresor", "nombre": "Turbocompresor", "sinonimos": ["turbo"]},
        {"id": "v8", "nombre": "Motor V8", "sinonimos": []},
    ]), encoding="utf-8")
    return ruta


@pytest.mark.parametrize("frase, esperado", [
    ("Freno a disco.", "Freno a disco"),
    ("freno a disco", "Freno a disco"),
    ("el disco de freno", "Freno a disco"),
    ("¡Che, el Turbo!", "Turbocompresor"),
    ("un turbo por favor", "Turbocompresor"),
    ("Motor V8.", "Motor V8"),
    ("CALIPER", "Freno a disco"),
])
def test_nombre_solo(catalogo, frase, esperado):
    r = atajo_catalogo.interpretar_atajo(frase, catalogo)
    assert r["accion"] == "mostrar" and r["objeto"] == esperado
    assert r["confianza"] >= 0.9   # no pide confirmación


@pytest.mark.parametrize("frase", [
    "mostrame el despiece del freno a disco",   # pide otra cosa: decide el motor de siempre
    "girá el turbo",
    "el freno desarmado",
    "freno",                                    # parte de un nombre: no alcanza
    "motor",
    "",
])
def test_no_es_solo_un_nombre(catalogo, frase):
    assert atajo_catalogo.interpretar_atajo(frase, catalogo) is None


def test_lee_el_catalogo_otra_vez_si_cambia(catalogo):
    assert atajo_catalogo.objeto_exacto("morsa de banco", catalogo) is None
    items = json.loads(catalogo.read_text(encoding="utf-8"))
    items.append({"id": "gen_morsa", "nombre": "Morsa de banco", "sinonimos": []})
    catalogo.write_text(json.dumps(items), encoding="utf-8")
    import os
    os.utime(catalogo, (catalogo.stat().st_atime, catalogo.stat().st_mtime + 5))
    assert atajo_catalogo.objeto_exacto("morsa de banco", catalogo) == "Morsa de banco"


def test_sin_catalogo(tmp_path):
    assert atajo_catalogo.interpretar_atajo("turbo", tmp_path / "no_hay.json") is None


def test_en_la_cadena(monkeypatch):
    # Con el catálogo real: no se llama a ningún motor
    def no_llamar(*a, **k):
        raise AssertionError("no tenía que consultar a un motor")
    monkeypatch.setattr(intent, "interpretar_llm", no_llamar)
    monkeypatch.setattr(intent, "_motor_suelto", no_llamar)
    r = intent.interpretar("Freno a disco.")
    assert (r["accion"], r["objeto"], r["motor"]) == ("mostrar", "Freno a disco", "catalogo")
    assert "atajo" in r["traza"]


def test_motor_elegido_no_usa_el_atajo(monkeypatch):
    # Si se pidió un motor ("usá Naive Bayes"), responde ese motor aunque la frase sea un nombre
    monkeypatch.setattr(intent, "_motor_suelto", lambda nombre: lambda f, traza: {**intent.DESCONOCIDO})
    assert intent.interpretar("Freno a disco.", motor="bayes")["motor"] == "bayes"
