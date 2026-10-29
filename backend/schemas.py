"""Contratos de datos (sección 7.4 de la propuesta)."""
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field

Accion = Literal["mostrar", "rotar", "escalar", "vista", "reiniciar", "desmontar", "explotar", "armar", "aislar", "conjunto",
                 "usar_motor", "consultar", "desconocido"]
Vista = Literal["frontal", "trasera", "lateral", "superior"]
# Comandos para manejar los modelos por voz ("usá el árbol para los gestos", "¿qué modelo respondió?").
# "automatico" vuelve a la cadena por defecto. Para los pedidos: los intérpretes del backend; para los gestos:
# los clasificadores del visor.
MOTORES_PEDIDOS = ("automatico", "claude", "consenso", "llm_propio", "transformer", "modelo", "bayes", "reglas")
MOTORES_GESTOS = ("automatico", "reglas", "arbol", "red")
Motor = Literal["automatico", "claude", "consenso", "llm_propio", "transformer", "modelo", "bayes", "reglas", "arbol",
                "red"]
Ambito = Literal["pedidos", "gestos"]
Pregunta = Literal["que_motor", "confianza", "explicar"]

LARGO_MAXIMO_FRASE = 300


class Pedido(BaseModel):
    frase: str = Field(min_length=1, max_length=LARGO_MAXIMO_FRASE)
    # Motor elegido por voz ("usá el LLM propio"); sin él, la cadena por defecto
    motor: Optional[Literal[MOTORES_PEDIDOS]] = None


class Parametros(BaseModel):
    vista: Optional[Vista] = None
    factor_escala: Optional[float] = None
    motor: Optional[Motor] = None          # usar_motor
    ambito: Optional[Ambito] = None        # usar_motor
    pregunta: Optional[Pregunta] = None    # consultar


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
    correccion: Optional[list[str]] = None   # errores de tipeo corregidos por la búsqueda difusa
    tiempo_ms: Optional[int] = None


# JSON Schema que se le exige al LLM (structured outputs). Es el mismo contrato que
# Intencion; los rangos (confianza 0-1, escala 0.2-5) los aplica safety.py.
ESQUEMA_INTENCION = {
    "type": "object",
    "properties": {
        "accion": {"type": "string", "enum": ["mostrar", "rotar", "escalar", "vista", "reiniciar", "desmontar",
                                               "explotar", "armar", "aislar", "conjunto", "usar_motor", "consultar",
                                               "desconocido"]},
        "objeto": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "parametros": {
            "type": "object",
            "properties": {
                "vista": {"anyOf": [{"type": "string", "enum": ["frontal", "trasera", "lateral", "superior"]},
                                    {"type": "null"}]},
                "factor_escala": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "motor": {"anyOf": [{"type": "string", "enum": list(get_args(Motor))}, {"type": "null"}]},
                "ambito": {"anyOf": [{"type": "string", "enum": list(get_args(Ambito))}, {"type": "null"}]},
                "pregunta": {"anyOf": [{"type": "string", "enum": list(get_args(Pregunta))}, {"type": "null"}]},
            },
            "required": ["vista", "factor_escala", "motor", "ambito", "pregunta"],
            "additionalProperties": False,
        },
        "confianza": {"type": "number"},
    },
    "required": ["accion", "objeto", "parametros", "confianza"],
    "additionalProperties": False,
}
