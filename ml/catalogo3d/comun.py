"""Rutas y descargas del catálogo 3D (Objaverse). Solo biblioteca estándar: lo usa también scripts/descargar_modelos3d.py.

Los datos grandes van en HOLOGRAMA_DATOS (por defecto ml/data/externo/, que no se versiona). Para pasarlos a otro
disco alcanza con mover la carpeta y definir la variable, por ejemplo HOLOGRAMA_DATOS=D:/holograma.
"""
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
DATOS = Path(os.getenv("HOLOGRAMA_DATOS") or RAIZ / "ml" / "data" / "externo")
OBJAVERSE = DATOS / "objaverse"
METADATOS = OBJAVERSE / "metadatos"
GLBS = OBJAVERSE / "glbs"
SELECCION = RAIZ / "ml" / "catalogo3d" / "seleccion.csv"
HF = "https://huggingface.co/datasets"


def descargar(url, destino, intentos=4):
    """Baja url a destino (vía .parte, para no dejar archivos a medias). Devuelve el tamaño en bytes."""
    destino = Path(destino)
    if destino.exists():
        return destino.stat().st_size
    destino.parent.mkdir(parents=True, exist_ok=True)
    parte = destino.with_name(destino.name + ".parte")
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tpo-holograma"}),
                                        timeout=60) as r, open(parte, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            parte.replace(destino)
            return destino.stat().st_size
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            if intento == intentos - 1:
                raise
            time.sleep(2 ** intento)
    return 0
