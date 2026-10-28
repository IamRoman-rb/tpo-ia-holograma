"""Contratos de datos (sección 7.4 de la propuesta)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Accion = Literal["mostrar", "rotar", "escalar", "vista", "reiniciar", "desmontar", "explotar", "armar", "desconocido"]
Vista = Literal["frontal", "trasera", "lateral", "superior"]

LARGO_MAXIMO_FRASE = 300


class Pedido(BaseModel):
    frase: str = Field(min_length=1, max_length=LARGO_MAXIMO_FRASE)


class Parametros(BaseModel):
    vista: Optional[Vista] = None
    factor_escala: Optional[float] = None


class Intencion(BaseModel):
    accion: Accion
    objeto: Optional[str] = None
    parametros: Parametros = Parametros()
    confianza: float = Field(ge=0, le=1)


class RespuestaObjeto(BaseModel):
    origen: Literal["catalogo", "generado"]
    id: str
    nombre: str
    modelo_url: str
    similitud: Optional[float] = None
    metodo: Optional[str] = None
    tiempo_ms: Optional[int] = None


# JSON Schema que se le exige al LLM (structured outputs). Es el mismo contrato que
# Intencion; los rangos (confianza 0-1, escala 0.2-5) los aplica safety.py.
ESQUEMA_INTENCION = {
    "type": "object",
    "properties": {
        "accion": {"type": "string", "enum": ["mostrar", "rotar", "escalar", "vista", "reiniciar", "desmontar", "explotar", "armar", "desconocido"]},
        "objeto": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "parametros": {
            "type": "object",
            "properties": {
                "vista": {"anyOf": [{"type": "string", "enum": ["frontal", "trasera", "lateral", "superior"]},
                                    {"type": "null"}]},
                "factor_escala": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
            "required": ["vista", "factor_escala"],
            "additionalProperties": False,
        },
        "confianza": {"type": "number"},
    },
    "required": ["accion", "objeto", "parametros", "confianza"],
    "additionalProperties": False,
}
