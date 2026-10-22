import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import generate3d
import intent
import main
import retrieval

MODELOS = Path(__file__).resolve().parents[2] / "frontend" / "models"
GLB = (MODELOS / "motor_electrico.glb").read_bytes()


@pytest.fixture
def entorno(monkeypatch, tmp_path):
    """Catálogo y carpeta de generados temporales + servidor de generación simulado."""
    catalogo = tmp_path / "catalogo.json"
    shutil.copy(MODELOS / "catalogo.json", catalogo)
    monkeypatch.setattr(generate3d, "RUTA_CATALOGO", catalogo)
    monkeypatch.setattr(generate3d, "DIR_GENERADOS", tmp_path / "generados")
    monkeypatch.setenv("GEN3D_MODO", "remoto")
    monkeypatch.setenv("GEN3D_URL", "http://generador.test")
    monkeypatch.setenv("GEN3D_TOKEN", "secreto")
    monkeypatch.setattr(generate3d, "prompt_en_ingles", lambda d: f"{d}, test")

    servidor = {"llamadas": 0, "respuesta": httpx.Response(200, content=GLB)}

    def responder(pedido):
        servidor["llamadas"] += 1
        assert pedido.headers["Authorization"] == "Bearer secreto"
        assert json.loads(pedido.content)["prompt"].endswith(", test")
        return servidor["respuesta"]

    monkeypatch.setattr(generate3d, "_cliente_http", httpx.Client(transport=httpx.MockTransport(responder)))
    return {"catalogo": catalogo, "dir": tmp_path / "generados", "servidor": servidor}


def test_glb_valido():
    assert generate3d.es_glb_valido(GLB)
    assert not generate3d.es_glb_valido(b"no es un modelo")
    assert not generate3d.es_glb_valido(GLB[:-4])            # largo declarado distinto del real
    assert not generate3d.es_glb_valido(b"glTF" + GLB[4:] + b"0" * generate3d.TAMANO_MAXIMO)


def test_identificador():
    assert generate3d.identificador("Bomba de agua centrífuga!") == "gen_bomba_de_agua_centrifuga"


def test_genera_guarda_y_reutiliza(entorno):
    r = generate3d.generar("bomba de agua")
    assert r["modelo_url"] == "models/generados/gen_bomba_de_agua.glb"
    assert (entorno["dir"] / "gen_bomba_de_agua.glb").read_bytes() == GLB
    item = json.loads(entorno["catalogo"].read_text(encoding="utf-8"))[-1]
    assert item["origen"] == "generado" and "No apto para fabricación" in item["descripcion"]

    generate3d.generar("bomba de agua")                       # segunda vez: no llama al servidor
    assert entorno["servidor"]["llamadas"] == 1
    assert sum(it["id"] == "gen_bomba_de_agua" for it in json.loads(entorno["catalogo"].read_text(encoding="utf-8"))) == 1


def test_archivo_invalido_no_se_guarda(entorno):
    entorno["servidor"]["respuesta"] = httpx.Response(200, content=b"<html>error</html>")
    with pytest.raises(generate3d.GeneracionFallida):
        generate3d.generar("bomba de agua")
    assert not (entorno["dir"] / "gen_bomba_de_agua.glb").exists()


def test_servidor_caido(entorno):
    entorno["servidor"]["respuesta"] = httpx.Response(500)
    with pytest.raises(generate3d.GeneracionFallida):
        generate3d.generar("bomba de agua")


def test_desactivada(monkeypatch):
    monkeypatch.setenv("GEN3D_MODO", "desactivado")
    with pytest.raises(generate3d.GeneracionNoDisponible):
        generate3d.generar("bomba de agua")


def test_api_genera_y_despues_lo_encuentra_en_el_catalogo(entorno, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "interpretar", lambda f: {**intent.interpretar_reglas(f), "motor": "reglas"})
    cat = retrieval.Catalogo(ruta_catalogo=entorno["catalogo"], ruta_indice=tmp_path / "i.json", usar_embeddings=False)
    monkeypatch.setattr(retrieval, "_catalogo", cat)
    main.limite._pedidos.clear()
    main.limite_generacion._pedidos.clear()
    cliente = TestClient(main.app)

    r = cliente.post("/object", json={"frase": "bomba de agua"}).json()
    assert r["origen"] == "generado" and r["id"] == "gen_bomba_de_agua"

    r = cliente.post("/object", json={"frase": "bomba de agua"}).json()
    assert r["origen"] == "generado" and r["metodo"] == "palabras"   # reutilizado desde el catálogo
    assert entorno["servidor"]["llamadas"] == 1

    assert cliente.post("/object", json={"frase": "una pistola"}).status_code == 422
