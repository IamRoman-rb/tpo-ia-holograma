"""Verifica que la normalización de Python sea idéntica a la del navegador.

Si difieren, el modelo se entrenaría con números distintos a los que ve en el visor.
Uso (desde la raíz):  python -m pytest ml
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preparar import COLUMNAS_PUNTOS, normalizar

RAIZ = Path(__file__).resolve().parent.parent


def fila_de_ejemplo():
    rng = np.random.default_rng(0)
    puntos = rng.uniform(0.2, 0.8, size=(21, 3))
    puntos[:, 2] = rng.uniform(-0.1, 0.1, size=21)
    datos = {"clase": "pellizco", "persona": "p0", "mano": "derecha", "aspecto": 640 / 480}
    datos.update(dict(zip(COLUMNAS_PUNTOS, puntos.reshape(-1))))
    return pd.DataFrame([datos]), puntos


@pytest.mark.skipif(shutil.which("node") is None, reason="node no está instalado")
def test_normalizacion_igual_que_el_navegador():
    df, puntos = fila_de_ejemplo()
    script = (
        "import { normalizar, aVector } from './frontend/js/normalize.js';"
        f"const p = {json.dumps([dict(x=x, y=y, z=z) for x, y, z in puntos.tolist()])};"
        f"console.log(JSON.stringify(aVector(normalizar(p, {640 / 480}))));"
    )
    salida = subprocess.run(["node", "--input-type=module", "-e", script], cwd=RAIZ,
                            capture_output=True, text=True, check=True).stdout
    js = np.array(json.loads(salida))
    np.testing.assert_allclose(normalizar(df)[0], js, atol=1e-5)


def test_normalizacion_invariante_a_posicion_y_escala():
    df, puntos = fila_de_ejemplo()
    movida = df.copy()
    for i in range(21):
        movida[f"x{i}"] = df[f"x{i}"] * 0.5 + 0.2
        movida[f"y{i}"] = df[f"y{i}"] * 0.5 + 0.1
        movida[f"z{i}"] = df[f"z{i}"] * 0.5
    np.testing.assert_allclose(normalizar(df), normalizar(movida), atol=1e-5)


@pytest.mark.skipif(shutil.which("node") is None, reason="node no está instalado")
def test_csv_de_la_captura_se_carga_bien(tmp_path):
    # Genera un CSV con el mismo código que usa frontend/captura.html
    script = """
import { CLASES, aCsv } from './frontend/js/dataset.js';
import { manoConVariaciones } from './frontend/js/simulador.js';
const muestras = [];
CLASES.forEach((c, i) => {
  for (let n = 0; n < 4; n++) {
    const { manos, lados } = manoConVariaciones(c.id, i * 10 + n);
    muestras.push({ clase: c.id, persona: n === 3 ? 'simulacion' : 'p01', mano: lados[0], aspecto: 640 / 480, puntos: manos[0] });
  }
});
console.log(JSON.stringify({ clases: CLASES.map((c) => c.id), csv: aCsv(muestras) }));
"""
    salida = subprocess.run(["node", "--input-type=module", "-e", script], cwd=RAIZ,
                            capture_output=True, text=True, check=True, encoding="utf-8").stdout
    datos = json.loads(salida)
    (tmp_path / "captura.csv").write_text(datos["csv"], encoding="utf-8")

    from preparar import CLASES, cargar
    assert datos["clases"] == CLASES, "las clases de la página y de Python no coinciden"
    df = cargar(tmp_path)
    assert len(df) == 15                      # las 5 filas "simulacion" se descartan
    assert set(df["clase"]) == set(CLASES)
    assert normalizar(df).shape == (15, 63)
    assert len(cargar(tmp_path, incluir_simulacion=True)) == 20
