"""Carga y normalización del dataset de gestos (secciones 12.2 y 12.3).

Formato de los CSV (frontend/captura.html o ml/collect_gestures.py):
    clase, persona, mano, aspecto, x0, y0, z0, ..., x20, y20, z20
con las coordenadas tal como las entrega MediaPipe.

La normalización es la MISMA que frontend/js/normalize.js (lo verifica test_preparar.py):
  1. x y z se multiplican por la relación de aspecto (MediaPipe normaliza x por el ancho
     e y por el alto: sin esto, una misma distancia mide distinto en cada eje).
  2. Origen en la muñeca (punto 0).
  3. Escala: distancia muñeca → nudillo del dedo medio (punto 9).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLASES = ["pellizco", "palma_abierta", "puño", "señalar", "ninguno"]
COLUMNAS_PUNTOS = [f"{e}{i}" for i in range(21) for e in "xyz"]
PERSONA_SIMULADA = "simulacion"


def cargar(directorio, incluir_simulacion=False) -> pd.DataFrame:
    """Une todos los CSV del directorio. Agrega 'archivo' y 'fila' para poder cruzar resultados."""
    archivos = sorted(Path(directorio).glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos CSV en {directorio}")
    partes = []
    for ruta in archivos:
        df = pd.read_csv(ruta, dtype={"persona": str, "mano": str})
        df.insert(0, "archivo", ruta.name)
        df.insert(1, "fila", range(len(df)))
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)

    faltan = {"clase", "persona", "aspecto", *COLUMNAS_PUNTOS} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(faltan)}")
    desconocidas = set(df["clase"]) - set(CLASES)
    if desconocidas:
        raise ValueError(f"Clases desconocidas en el dataset: {sorted(desconocidas)}")
    if not incluir_simulacion:
        df = df[df["persona"] != PERSONA_SIMULADA].reset_index(drop=True)
    return df


def normalizar(df: pd.DataFrame) -> np.ndarray:
    """Devuelve una matriz (n, 63) con los landmarks normalizados: la entrada del MLP."""
    p = df[COLUMNAS_PUNTOS].to_numpy(dtype=np.float64, copy=True).reshape(-1, 21, 3)
    aspecto = df["aspecto"].to_numpy(dtype=np.float64)[:, None]
    p[:, :, 0] *= aspecto
    p[:, :, 2] *= aspecto
    p -= p[:, :1, :]                                   # origen en la muñeca
    escala = np.linalg.norm(p[:, 9, :], axis=1)        # muñeca → nudillo medio
    escala[escala == 0] = 1
    return (p / escala[:, None, None]).reshape(-1, 63).astype(np.float32)
