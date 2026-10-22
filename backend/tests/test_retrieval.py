import pytest

from retrieval import Catalogo


@pytest.fixture(scope="module")
def palabras(tmp_path_factory):
    return Catalogo(ruta_indice=tmp_path_factory.mktemp("idx") / "index.json", usar_embeddings=False)


@pytest.mark.parametrize("consulta, esperado", [
    ("valvula de acero", "valvula_compuerta"),
    ("rulemán", "rodamiento_bolas"),
    ("rueda dentada", "engranaje_recto"),
    ("tornillos", "tornillo_hexagonal"),
    ("motor", "motor_electrico"),
    ("silla de oficina", None),
])
def test_busqueda_por_palabras(palabras, consulta, esperado):
    r = palabras.buscar(consulta)
    assert (r["id"] if r else None) == esperado


def test_embeddings(tmp_path):
    pytest.importorskip("sentence_transformers")
    cat = Catalogo(ruta_indice=tmp_path / "index.json")
    assert cat.metodo == "hibrido"
    # nombre conocido: lo resuelven las palabras clave
    assert cat.buscar("llave de paso")["metodo"] == "palabras"
    # paráfrasis sin palabras en común: la resuelven los embeddings
    r = cat.buscar("máquina que convierte electricidad en movimiento")
    assert r["id"] == "motor_electrico" and r["metodo"] == "embeddings"
    assert cat.buscar("zapatilla deportiva") is None
    # palabras sueltas que no están en el catálogo: no se usan embeddings
    assert cat.buscar("mesa") is None and cat.buscar("silla") is None
    assert cat.buscar("balero")["id"] == "rodamiento_bolas"   # sinónimo: lo resuelven las palabras
    assert (tmp_path / "index.json").exists()
