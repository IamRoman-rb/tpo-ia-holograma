import pytest
from fastapi.testclient import TestClient

import intent
import main
import retrieval


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    # Sin red: intención por reglas, búsqueda por palabras y sin generación 3D (aunque el .env tenga una clave)
    monkeypatch.setattr(main, "interpretar", lambda f, motor=None: {**intent.interpretar_reglas(f), "motor": "reglas"})
    monkeypatch.setenv("GEN3D_MODO", "desactivado")
    cat = retrieval.Catalogo(ruta_indice=tmp_path / "index.json", usar_embeddings=False)
    monkeypatch.setattr(retrieval, "_catalogo", cat)
    main.limite._pedidos.clear()
    return TestClient(main.app)


def test_health(cliente):
    assert cliente.get("/health").json()["estado"] == "ok"


def test_intent(cliente):
    r = cliente.post("/intent", json={"frase": "mostrame un rulemán desde arriba"}).json()
    assert r["accion"] == "mostrar" and r["parametros"]["vista"] == "superior" and r["motor"] == "reglas"
    # La traza trae la respuesta antes de validar, para el panel de actividad
    assert r["traza"]["crudo"]["accion"] == "mostrar"


def test_object_del_catalogo(cliente):
    r = cliente.post("/object", json={"frase": "ruleman"}).json()
    assert r["origen"] == "catalogo" and r["modelo_url"] == "models/rodamiento_bolas.glb"


def test_object_inexistente_404(cliente):
    assert cliente.post("/object", json={"frase": "silla de oficina"}).status_code == 404


def test_frase_demasiado_larga(cliente):
    assert cliente.post("/intent", json={"frase": "x" * 301}).status_code == 422


def test_catalogo(cliente):
    assert len(cliente.get("/catalog").json()) == 5
