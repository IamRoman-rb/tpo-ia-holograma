"""Atajo para los pedidos que son solo el nombre de un objeto del catálogo: "Freno a disco", "el turbo", "Motor V8.".

Sin verbo, los motores estadísticos adivinan la acción por las palabras del objeto. En las frases de entrenamiento,
los pedidos de despiece se concentran en piezas mecánicas ("despiece del embrague") y los de mostrar se reparten
entre miles de objetos, así que a Naive Bayes un nombre mecánico solo le suena a despiece. Si la frase entera es el
nombre o un sinónimo de algo del catálogo, no hay nada que adivinar: el pedido es mostrarlo.

Solo cuenta la coincidencia exacta (sin tildes, mayúsculas, puntuación, artículos ni muletillas). Cualquier otra
palabra ("el freno desarmado", "girá el turbo") deja la frase a los motores de siempre.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from texto import normalizar

RUTA_CATALOGO = Path(__file__).resolve().parent.parent / "frontend" / "models" / "catalogo.json"
CONFIANZA = 0.95   # por encima del umbral de confirmación: es el objeto tal cual está en el catálogo

# Lo que se puede decir antes o después del nombre sin cambiar el pedido
RELLENO_INICIO = re.compile(r"^(?:(?:che|dale|bueno|ok|hola|eh|oye|a ver|ahora)\s+)*(?:(?:un|una|unos|unas|el|la|los|las)\s+)?")
RELLENO_FIN = re.compile(r"(?:\s+(?:por favor|porfa|dale|ya|ahora|gracias))*$")


def limpiar(frase: str) -> str:
    """La frase sin tildes, puntuación, muletillas ni artículo inicial: "¡Che, el Turbo!" → "turbo"."""
    t = re.sub(r"[.,]", " ", normalizar(frase))
    t = re.sub(r"\s+", " ", t).strip()
    return RELLENO_FIN.sub("", RELLENO_INICIO.sub("", t)).strip()


@lru_cache(maxsize=4)
def _nombres(ruta: str, version: float) -> dict:
    """{nombre limpio: nombre del catálogo}. `version` es la fecha del archivo: un objeto generado nuevo entra solo."""
    items = json.loads(Path(ruta).read_text(encoding="utf-8"))
    nombres = {}
    for it in items:
        for variante in [it["nombre"], *it.get("sinonimos", [])]:
            clave = limpiar(variante)
            if clave:
                nombres.setdefault(clave, it["nombre"])   # ante un sinónimo repetido, gana el primero del catálogo
    return nombres


def objeto_exacto(frase: str, ruta: Path = RUTA_CATALOGO) -> str | None:
    """El nombre del catálogo si la frase es justo ese objeto; si no, None."""
    if not ruta.exists():
        return None
    return _nombres(str(ruta), ruta.stat().st_mtime).get(limpiar(frase))


def interpretar_atajo(frase: str, ruta: Path = RUTA_CATALOGO) -> dict | None:
    """La intención de mostrar ese objeto, o None si la frase no es solo un nombre del catálogo."""
    nombre = objeto_exacto(frase, ruta)
    if not nombre:
        return None
    return {"accion": "mostrar", "objeto": nombre, "parametros": {"vista": None, "factor_escala": None},
            "confianza": CONFIANZA}
