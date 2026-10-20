"""Contratos de datos (sección 7.4 de la propuesta)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Pedido(BaseModel):
    frase: str = Field(min_length=1, max_length=300)


class Parametros(BaseModel):
    vista: Optional[Literal["frontal", "trasera", "lateral", "superior"]] = None
    factor_escala: Optional[float] = None


class Intencion(BaseModel):
    accion: Literal["mostrar", "rotar", "escalar", "vista", "reiniciar", "desconocido"]
    objeto: Optional[str] = None
    parametros: Parametros = Parametros()
    confianza: float = Field(ge=0, le=1)


class RespuestaObjeto(BaseModel):
    origen: Literal["catalogo", "generado"]
    modelo_url: str
    similitud: Optional[float] = None
    tiempo_ms: Optional[int] = None
