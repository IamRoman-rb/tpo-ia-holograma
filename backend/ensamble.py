"""Generación de objetos POR PIEZAS con el LLM (técnica 4, IA generativa).

Los generadores 3D de imagen o texto (TripoSR, Shap-E) devuelven una sola malla fundida:
no se puede desarmar. Para poder desmontar el objeto pieza por pieza, el LLM lo diseña como
un ensamble: una lista de piezas reales (cuerpo, tapa, eje, tornillos…), cada una construida
con formas básicas, con su orden y su dirección de desmontaje.

Flujo:  descripción → LLM (JSON Schema) → validación (Pydantic + límites) → geometría → .glb por piezas.
"""
import json
import logging
import math
from pathlib import Path
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError, field_validator

import geometria as g
from intent import MODELO, MODELOS_CON_FALLBACK

log = logging.getLogger(__name__)

PROMPT_SISTEMA = (Path(__file__).parent / "prompts" / "ensamble_system.txt").read_text(encoding="utf-8")

FORMAS = ["cilindro", "cono", "tubo", "prisma_hexagonal", "engranaje", "caja", "esfera", "toroide"]
MATERIALES = {
    "acero": ((0.80, 0.80, 0.82), 1.0, 0.25),
    "hierro": ((0.25, 0.27, 0.30), 0.7, 0.45),
    "aluminio": ((0.85, 0.86, 0.88), 0.9, 0.35),
    "laton": ((0.78, 0.60, 0.25), 0.9, 0.30),
    "cobre": ((0.72, 0.42, 0.20), 0.9, 0.35),
    "plastico": ((0.15, 0.15, 0.17), 0.0, 0.60),
    "goma": ((0.08, 0.08, 0.08), 0.0, 0.90),
    "pintado": ((0.10, 0.30, 0.60), 0.3, 0.50),
}
MAX_PIEZAS, MAX_FORMAS = 30, 12
MEDIDA_MIN, MEDIDA_MAX, POSICION_MAX = 0.05, 300.0, 1000.0

_nulo_numero = {"anyOf": [{"type": "number"}, {"type": "null"}]}
ESQUEMA_ENSAMBLE = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "descripcion": {"type": "string"},
        "piezas": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "nombre": {"type": "string"},
                "descripcion": {"type": "string"},
                "material": {"type": "string", "enum": list(MATERIALES)},
                "color": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "orden_desmontaje": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "direccion_desmontaje": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
                "formas": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "tipo": {"type": "string", "enum": FORMAS},
                        "medidas": {
                            "type": "object",
                            "properties": {k: _nulo_numero for k in (
                                "radio", "radio_superior", "radio_interior", "radio_tubo",
                                "alto", "ancho", "profundidad", "dientes")},
                            "required": ["radio", "radio_superior", "radio_interior", "radio_tubo",
                                         "alto", "ancho", "profundidad", "dientes"],
                            "additionalProperties": False,
                        },
                        "posicion": {"type": "array", "items": {"type": "number"}},
                        "rotacion": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["tipo", "medidas", "posicion", "rotacion"],
                    "additionalProperties": False,
                }},
            },
            "required": ["id", "nombre", "descripcion", "material", "color", "orden_desmontaje",
                         "direccion_desmontaje", "formas"],
            "additionalProperties": False,
        }},
    },
    "required": ["nombre", "descripcion", "piezas"],
    "additionalProperties": False,
}


class EnsambleInvalido(Exception):
    pass


# ------------------------------------------------------------------ validación

def _vector3(v, maximo):
    if v is None or len(v) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        raise ValueError("se esperaba un vector de 3 números")
    return [max(-maximo, min(maximo, float(x))) for x in v]


class Medidas(BaseModel):
    radio: Optional[float] = None
    radio_superior: Optional[float] = None
    radio_interior: Optional[float] = None
    radio_tubo: Optional[float] = None
    alto: Optional[float] = None
    ancho: Optional[float] = None
    profundidad: Optional[float] = None
    dientes: Optional[float] = None


REQUERIDAS = {
    "cilindro": ("radio", "alto"), "cono": ("radio", "alto"), "tubo": ("radio", "radio_interior", "alto"),
    "prisma_hexagonal": ("radio", "alto"), "engranaje": ("radio", "alto", "dientes"),
    "caja": ("ancho", "alto", "profundidad"), "esfera": ("radio",), "toroide": ("radio", "radio_tubo"),
}


class Forma(BaseModel):
    tipo: Literal["cilindro", "cono", "tubo", "prisma_hexagonal", "engranaje", "caja", "esfera", "toroide"]
    medidas: Medidas
    posicion: list[float]
    rotacion: list[float]

    @field_validator("posicion")
    @classmethod
    def _posicion(cls, v):
        return _vector3(v, POSICION_MAX)

    @field_validator("rotacion")
    @classmethod
    def _rotacion(cls, v):
        return _vector3(v, 360.0)

    def validar_medidas(self):
        m = self.medidas
        for nombre in REQUERIDAS[self.tipo]:
            valor = getattr(m, nombre)
            if valor is None or not math.isfinite(valor) or valor < MEDIDA_MIN or valor > MEDIDA_MAX:
                raise ValueError(f"{self.tipo}: medida '{nombre}' fuera de rango ({valor})")
        if self.tipo == "tubo" and m.radio_interior >= m.radio:
            raise ValueError("tubo: el radio interior debe ser menor que el exterior")
        if self.tipo == "toroide" and m.radio_tubo >= m.radio:
            raise ValueError("toroide: el radio del tubo debe ser menor que el radio")
        if self.tipo == "engranaje" and not 6 <= m.dientes <= 120:
            raise ValueError("engranaje: entre 6 y 120 dientes")


class Pieza(BaseModel):
    id: str = Field(min_length=1, max_length=60)
    nombre: str = Field(min_length=1, max_length=80)
    descripcion: str = Field(max_length=300)
    material: Literal["acero", "hierro", "aluminio", "laton", "cobre", "plastico", "goma", "pintado"]
    color: Optional[str] = None
    orden_desmontaje: Optional[int] = None
    direccion_desmontaje: Optional[list[float]] = None
    formas: list[Forma]

    @field_validator("color")
    @classmethod
    def _color(cls, v):
        if v is None:
            return None
        v = v.strip()
        return v if len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in v[1:]) else None

    @field_validator("direccion_desmontaje")
    @classmethod
    def _direccion(cls, v):
        if v is None:
            return None
        v = _vector3(v, 1e6)
        largo = math.hypot(*v)
        return [x / largo for x in v] if largo > 1e-6 else None


class Ensamble(BaseModel):
    nombre: str = Field(min_length=1, max_length=120)
    descripcion: str = Field(max_length=500)
    piezas: list[Pieza]


def validar(datos: dict) -> Ensamble:
    """Valida el ensamble. Las formas inválidas se descartan; si no queda ninguna pieza, es inválido."""
    try:
        ensamble = Ensamble.model_validate(datos)
    except ValidationError as e:
        raise EnsambleInvalido(f"el ensamble no cumple el esquema: {e.error_count()} errores") from e

    piezas, ids = [], set()
    for pieza in ensamble.piezas[:MAX_PIEZAS]:
        formas = []
        for forma in pieza.formas[:MAX_FORMAS]:
            try:
                forma.validar_medidas()
                formas.append(forma)
            except ValueError as e:
                log.info("Forma descartada en '%s': %s", pieza.nombre, e)
        if not formas:
            continue
        id_ = pieza.id
        while id_ in ids:  # ids repetidos: se desambiguan
            id_ += "_b"
        ids.add(id_)
        piezas.append(pieza.model_copy(update={"id": id_, "formas": formas}))
    if not piezas:
        raise EnsambleInvalido(ensamble.descripcion or "el ensamble no tiene piezas válidas")
    return ensamble.model_copy(update={"piezas": piezas})


# ------------------------------------------------------------------ geometría

def _geometria(forma: Forma):
    m = forma.medidas
    if forma.tipo == "cilindro":
        return g.cilindro(m.radio, m.alto)
    if forma.tipo == "cono":
        return g.cono(m.radio, max(0.0, min(m.radio_superior or 0.0, MEDIDA_MAX)), m.alto)
    if forma.tipo == "tubo":
        return g.prisma_anular(g.circulo(m.radio), m.radio_interior, m.alto)
    if forma.tipo == "prisma_hexagonal":
        return g.cilindro(m.radio, m.alto, 6, plano=True)
    if forma.tipo == "engranaje":
        dientes = int(round(m.dientes))
        r_raiz = m.radio * (1 - min(0.25, 2.5 / dientes))
        agujero = m.radio_interior if m.radio_interior and m.radio_interior < r_raiz * 0.9 else r_raiz * 0.25
        return g.prisma_anular(g.perfil_engranaje(dientes, r_raiz, m.radio), agujero, m.alto)
    if forma.tipo == "caja":
        return g.caja(m.ancho, m.alto, m.profundidad)
    if forma.tipo == "esfera":
        return g.esfera(m.radio)
    return g.toroide(m.radio, m.radio_tubo)


def a_modelo(ensamble: Ensamble) -> g.Modelo:
    materiales = dict(MATERIALES)
    modelo = g.Modelo(materiales)
    for pieza in ensamble.piezas:
        material = pieza.material
        if pieza.color:  # color propio: se crea una variante del material
            rgb = tuple(int(pieza.color[i:i + 2], 16) / 255 for i in (1, 3, 5))
            material = f"{pieza.material}_{pieza.color[1:].lower()}"
            materiales[material] = (rgb, *MATERIALES[pieza.material][1:])
        modelo.pieza(pieza.id, pieza.nombre, pieza.descripcion, pieza.orden_desmontaje, pieza.direccion_desmontaje)
        for forma in pieza.formas:
            modelo.agregar(material, g.mover(_geometria(forma), forma.posicion, g.rotacion_grados(*forma.rotacion)))
    return modelo


# ------------------------------------------------------------------ LLM

def pedir_ensamble(descripcion: str, cliente=None) -> dict:
    """Le pide al LLM el ensamble del objeto. Devuelve el JSON sin validar."""
    cliente = cliente or anthropic.Anthropic(timeout=180.0, max_retries=1)
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if MODELO in MODELOS_CON_FALLBACK else {}
    r = cliente.beta.messages.create(
        model=MODELO,
        max_tokens=16000,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": descripcion}],
        # esfuerzo medio: diseñar la geometría de un ensamble necesita algo de razonamiento
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": ESQUEMA_ENSAMBLE}},
        **extra,
    )
    if r.stop_reason == "refusal":
        raise EnsambleInvalido("el modelo rechazó el pedido")
    if r.stop_reason == "max_tokens":
        raise EnsambleInvalido("la respuesta se cortó por largo")
    texto = next((b.text for b in r.content if b.type == "text"), None)
    if texto is None:
        raise EnsambleInvalido("la respuesta no trae texto")
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        raise EnsambleInvalido(f"JSON inválido: {e}") from e


def generar_glb(descripcion: str, cliente=None) -> tuple[bytes, Ensamble]:
    """Descripción → ensamble validado → .glb con una pieza por nodo."""
    ensamble = validar(pedir_ensamble(descripcion, cliente))
    return g.a_glb(a_modelo(ensamble), ensamble.nombre, unidad="cm"), ensamble
