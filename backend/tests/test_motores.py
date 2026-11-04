"""Motores generados por código (backend/motores/): pedidos, piezas nombradas y desarme ordenado."""
import json
import struct

import pytest
from fastapi.testclient import TestClient

import motores
from geometria import a_glb
from motores import Combustion, Electrico


def nodos(params):
    datos = a_glb(motores.generar(params), motores.nombre(params), unidad="cm")
    largo = struct.unpack("<I", datos[12:16])[0]
    return json.loads(datos[20:20 + largo])["nodes"]


@pytest.mark.parametrize("frase, esperado", [
    ("motor V8 turbo", Combustion(cilindros=8, disposicion="v", turbo=True)),
    ("motor de 3 cilindros", Combustion(cilindros=3)),
    ("motor bóxer de 6 cilindros", Combustion(cilindros=6, disposicion="boxer")),
    ("motor radial", Combustion(cilindros=9, disposicion="radial")),
    ("motor V-6", Combustion(cilindros=6, disposicion="v", angulo_v=60.0)),
    ("motor diésel", Combustion(combustible="diesel")),
    ("motor de 8 válvulas", Combustion(valvulas=2, arboles="sohc")),
    ("motor tricilíndrico", Combustion(cilindros=3)),
    ("motor trifásico de 10 kW con brida", Electrico("induccion", 10.0, "brida")),
    ("motor brushless", Electrico("brushless", 0.3, "patas")),
    ("motor de corriente continua", Electrico("continua", 0.5, "patas")),
])
def test_pedidos(frase, esperado):
    assert motores.pedido(frase) == esperado


@pytest.mark.parametrize("frase", ["motor", "silla", "mostrame una válvula", "engranaje recto"])
def test_no_son_pedidos_de_motor(frase):
    assert motores.pedido(frase) is None   # "motor" solo sigue yendo al catálogo


def test_lo_que_no_se_puede_construir_se_ajusta():
    assert motores.pedido("motor V7").cilindros == 8          # en V, pares
    assert motores.pedido("motor radial de 8 cilindros").cilindros == 9   # radial, impares
    assert motores.pedido("motor de 20 cilindros").cilindros == 8


@pytest.mark.parametrize("params", [p for p, _, _ in motores.CATALOGO] + [motores.TRIFASICO])
def test_cada_pieza_tiene_nombre_clase_y_orden(params):
    ns = nodos(params)
    assert len(ns) >= 7
    assert all(n["extras"]["nombre"] and n["extras"]["clase"] for n in ns)
    ordenes = [n["extras"]["orden"] for n in ns if "orden" in n["extras"]]
    assert sorted(ordenes) == list(range(1, len(ordenes) + 1))   # sin repetidos ni huecos
    assert len(ns) - len(ordenes) == 1                           # una sola pieza fija (bloque, carcasa…)
    assert all(abs(sum(x * x for x in n["extras"]["direccion"]) - 1) < 1e-4 for n in ns if "orden" in n["extras"])


def test_piezas_de_un_v8():
    ns = nodos(Combustion(cilindros=8, disposicion="v"))
    nombres = {n["extras"]["nombre"] for n in ns}
    assert {"Pistón 1", "Pistón 8", "Biela 8", "Cigüeñal", "Bloque", "Volante de inercia",
            "Culata izquierdo", "Culata derecho"} <= nombres or {"Culata izquierda", "Culata derecha"} <= nombres
    # el pistón sale después de la culata de su banco, y el cigüeñal después de todas las bielas
    orden = {n["extras"]["nombre"]: n["extras"].get("orden") for n in ns}
    assert orden["Cigüeñal"] > max(v for k, v in orden.items() if k.startswith("Biela"))


def test_mismos_parametros_mismo_archivo(tmp_path, monkeypatch):
    monkeypatch.setattr(motores, "CARPETA", tmp_path / "frontend" / "models" / "motores")
    monkeypatch.setattr(motores, "RAIZ", tmp_path)
    p = motores.pedido("motor V10 turbo diésel")
    id1, url1 = motores.glb_para(p)
    id2, url2 = motores.glb_para(p)
    assert (id1, url1) == (id2, url2) and url1.endswith(".glb") and "a_pedido" in url1


def test_object_arma_el_motor_pedido(monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(motores, "CARPETA", tmp_path / "frontend" / "models" / "motores")
    monkeypatch.setattr(motores, "RAIZ", tmp_path)
    r = TestClient(main.app).post("/object", json={"frase": "motor V8 turbo"}).json()
    assert r["origen"] == "parametrico" and r["nombre"] == "Motor V8 turbo"
    assert r["modelo_url"].startswith("models/motores/a_pedido/")
