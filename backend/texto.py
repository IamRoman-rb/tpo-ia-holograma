"""Utilidades de texto compartidas por el intérprete por reglas y la búsqueda por palabras."""
import re
import unicodedata


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin signos de puntuación."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s.,]", " ", t)).strip()
