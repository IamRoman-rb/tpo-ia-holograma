"""Catálogo de Objaverse con datos falsos: palabras, prioridad del catálogo propio, solo modelos bajados y desempate."""
import json

import numpy as np
import pytest

from retrieval import Catalogo, CatalogoObjaverse


def item(uid, nombre, sinonimos=(), piezas=None, calidad=1.0):
    return {"id": uid, "nombre": nombre, "sinonimos": list(sinonimos), "nombre_original": nombre,
            "descripcion": "", "categoria": "x", "piezas": piezas, "calidad": calidad, "tamano": 1,
            "atribucion": {"titulo": nombre, "autor": "Autora", "autor_url": "", "licencia": "CC BY 4.0",
                           "url": f"https://sketchfab.com/3d-models/{uid}"}}


@pytest.fixture
def objaverse(tmp_path):
    items = [item("v8", "Motor de auto", ["motor v8"], piezas=12), item("v8b", "Motor de auto", ["motor v8"], piezas=1),
             item("gui", "Guitarra"), item("mesa", "Mesa"), item("falta", "Heladera")]
    (tmp_path / "items.json").write_text(json.dumps(items), encoding="utf-8")
    glbs = tmp_path / "glbs"
    glbs.mkdir()
    for uid in ("v8", "v8b", "gui", "mesa"):   # la heladera no está bajada
        (glbs / f"{uid}.glb").write_bytes(b"glTF")
    return CatalogoObjaverse(tmp_path / "items.json", tmp_path / "no_hay.npz", glbs)


@pytest.fixture
def catalogo(tmp_path, objaverse):
    return Catalogo(ruta_indice=tmp_path / "index.json", usar_embeddings=False, objaverse=objaverse)


def test_solo_cuentan_los_bajados(objaverse, catalogo):
    assert len(objaverse) == 4
    assert catalogo.buscar("heladera") is None


@pytest.mark.parametrize("consulta, esperado", [
    ("guitarra", "gui"),
    ("una mesa", "mesa"),
    ("motor de auto", "v8"),          # nombra más que el "motor" del catálogo propio, y tiene más piezas que v8b
    ("motor", "motor_electrico"),     # a igual coincidencia, el catálogo propio va primero
    ("rulemán", "rodamiento_bolas"),
])
def test_prioridades(catalogo, consulta, esperado):
    assert catalogo.buscar(consulta)["id"] == esperado


def test_resultado_con_atribucion(catalogo):
    r = catalogo.buscar("guitarra")
    assert r["origen"] == "objaverse" and r["modelo_url"] == "/modelos3d/gui.glb"
    assert r["atribucion"]["licencia"] == "CC BY 4.0" and r["atribucion"]["autor"] == "Autora"


def test_corrector_conoce_las_palabras_de_objaverse(catalogo):
    assert catalogo.buscar("guitara")["id"] == "gui"


def test_embeddings_eligen_el_mas_parecido(tmp_path, objaverse):
    # 3 textos: el de "gui" apunta a la consulta; v8 y v8b empatan en otro sentido
    objaverse.vectores = np.array([[1, 0], [0, 1], [0.6, 0.8]], np.float32)
    objaverse.filas_item, objaverse.filas_texto = np.array([0, 1, 2]), np.array([0, 0, 1])
    i, sim = objaverse.mejor_por_embeddings(np.array([0, 1], np.float32))
    assert objaverse.items[i]["id"] == "gui" and sim == pytest.approx(1.0)
    i, _ = objaverse.mejor_por_embeddings(np.array([1, 0], np.float32))
    assert objaverse.items[i]["id"] == "v8"   # empata con v8b; gana el que se puede desarmar
