"""Interpretación de la intención: frase libre -> dict con el esquema de la sección 14.1.A.

Dos motores:
  - "llm": Claude con salida estructurada (JSON Schema). Es el motor principal.
  - "reglas": coincidencia de palabras clave. Se usa cuando no hay credenciales, no hay
    internet o la API falla, para que la demo nunca dependa de la red. También sirve como
    línea base para comparar en eval/eval_intent.py (sección 5.3: "alternativas").
"""
import json
import logging
import os
import re
from pathlib import Path

import anthropic

from schemas import ESQUEMA_INTENCION
from texto import normalizar

log = logging.getLogger(__name__)

PROMPT_SISTEMA = (Path(__file__).parent / "prompts" / "intent_system.txt").read_text(encoding="utf-8")
MODELO = os.getenv("LLM_MODEL") or "claude-opus-5-5"
# Si el modelo rechaza un pedido por seguridad, la API lo reintenta en otro modelo
# dentro de la misma llamada. Solo algunos modelos aceptan este parámetro.
MODELOS_CON_FALLBACK = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

DESCONOCIDO = {"accion": "desconocido", "objeto": None,
               "parametros": {"vista": None, "factor_escala": None}, "confianza": 0.0}

_cliente = None


class RespuestaInvalida(Exception):
    pass


def _cliente_llm() -> anthropic.Anthropic:
    global _cliente
    if _cliente is None:
        # Timeout corto: si el LLM tarda, conviene caer a las reglas antes que congelar la demo.
        _cliente = anthropic.Anthropic(timeout=15.0, max_retries=1)
    return _cliente


def interpretar(frase: str) -> dict:
    """Devuelve la intención (sin validar) e indica qué motor la produjo en "motor"."""
    try:
        return {**interpretar_llm(frase), "motor": "llm"}
    except (anthropic.AnthropicError, RespuestaInvalida, TypeError) as e:
        # TypeError: el SDK no encontró credenciales.
        log.warning("LLM no disponible (%s: %s); se usan las reglas", type(e).__name__, e)
        return {**interpretar_reglas(frase), "motor": "reglas"}


# ------------------------------------------------------------------- motor LLM

def interpretar_llm(frase: str, cliente=None) -> dict:
    cliente = cliente or _cliente_llm()
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if MODELO in MODELOS_CON_FALLBACK else {}
    r = cliente.beta.messages.create(
        model=MODELO,
        max_tokens=2048,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": frase}],
        # effort bajo: es una clasificación corta, importa más la latencia que razonar a fondo
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ESQUEMA_INTENCION}},
        **extra,
    )
    if r.stop_reason == "refusal":
        return dict(DESCONOCIDO)
    if r.stop_reason == "max_tokens":
        raise RespuestaInvalida("respuesta cortada por max_tokens")
    texto = next((b.text for b in r.content if b.type == "text"), None)
    if texto is None:
        raise RespuestaInvalida("la respuesta no trae texto")
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        raise RespuestaInvalida(f"JSON inválido: {e}") from e


# ------------------------------------------------------------------- motor de reglas

VERBOS_MOSTRAR = (r"(?:por favor\s+)?(?:mostrame|mostrar|mostra|muestrame|muestra|quiero ver|quisiera ver|"
                  r"quiero|necesito ver|necesito|traeme|trae|dame|cargame|carga|cargar|busca|buscame|"
                  r"abri|abrime|poneme|pone|ver|genera|generame|crea|creame|haceme|hace)")
ARTICULOS = r"(?:un|una|unos|unas|el|la|los|las|otro|otra)"

_PREFIJO_VISTA = r"(?:(?:desde|de|por|visto|vista) )?(?:la )?(?:(?:vista|parte) )?(?:de )?"
VISTAS = [
    ("trasera", _PREFIJO_VISTA + r"(?:atras|detras|trasera|espalda)"),
    ("frontal", _PREFIJO_VISTA + r"(?:adelante|el frente|frente|frontal|delantera)"),
    ("lateral", _PREFIJO_VISTA + r"(?:el costado|costado|un lado|el lado|lado|lateral|perfil)"),
    ("superior", _PREFIJO_VISTA + r"(?:arriba|encima|superior|planta)"),
]

ESCALAS = [
    (r"\b(\d+(?:[.,]\d+)?)\s*veces\b", None),
    (r"\bel doble\b|\bdoble\b|\bduplica", 2.0),
    (r"\bel triple\b|\btriple\b|\btriplica", 3.0),
    (r"\bla mitad\b|\bmitad\b", 0.5),
    (r"\bagrand|\bmas grande|\baument|\bamplia|\bzoom\b|\bacerca", 1.5),
    (r"\bachic|\bmas chic|\bmas pequen|\breduc|\bdisminu|\baleja", 0.67),
]


def _factor(t):
    for patron, valor in ESCALAS:
        m = re.search(patron, t)
        if m:
            return float(m.group(1).replace(",", ".")) if valor is None else valor
    return None


def _vista(t):
    for nombre, patron in VISTAS:
        if re.search(rf"\b{patron}\b", t):
            return nombre
    return None


def _objeto(t):
    m = re.match(rf"^{VERBOS_MOSTRAR}\b\s*", t)
    if m:
        resto = t[m.end():]
    elif re.match(rf"^{ARTICULOS}\s+", t):
        resto = t  # "una válvula de acero": un objeto sin verbo también es un pedido de mostrar
    else:
        return None
    # sacar las frases de vista ("desde atrás", "la parte de atrás") que no describen al objeto
    for _, patron in VISTAS:
        resto = re.sub(rf"\b{patron}\b", " ", resto)
    # cortar en lo que ya es otra acción o una cortesía
    resto = re.split(r"\b(?:y|hacelo|ponelo|agrandalo|achicalo|giralo|rotalo|por favor)\b|[.,]", resto)[0]
    resto = re.sub(rf"^{ARTICULOS}\s+", "", resto.strip()).strip()
    return resto or None


# Desarmado por piezas. El orden importa: "desarmalo" contiene "armalo".
RE_EXPLOTAR = (r"\bexplot|\bdespiece|\bdesarmad[oa]s?\b|\bdesmontad[oa]s?\b"
               r"|\b(?:todas las|las) (?:piezas|partes)\b|\bdesarm\w* (?:todo|entero|completo|del todo)")
RE_DESMONTAR = r"\bdesarm|\bdesmont|\bsac\w* (?:una |la |otra |la siguiente |la proxima )?(?:pieza|parte)|\bsepar"
RE_ARMAR = (r"\b(?:arma|armalo|armala|armar|armalos|monta|montalo|montala|montar|reensambla\w*|ensambla\w*)\b"
            r"|\bvolve a armar|\bjunta (?:las )?piezas")
PALABRAS_DESPIECE = r"\b(?:el |la )?(?:despiece|piezas|partes|desarmad[oa]s?|desmontad[oa]s?|explotad[oa]s?)\b"


def interpretar_reglas(frase: str) -> dict:
    t = normalizar(frase)
    objeto, vista, factor = _objeto(t), _vista(t), _factor(t)

    if re.search(RE_EXPLOTAR, t):
        accion = "explotar"
        # "mostrame la válvula desarmada": el objeto es la válvula, no "válvula desarmada"
        if objeto:
            objeto = re.sub(r"^(?:de |del )", "", re.sub(PALABRAS_DESPIECE, " ", objeto).strip()).strip()
            objeto = re.sub(rf"^{ARTICULOS}\s+", "", objeto).strip() or None
    elif re.search(RE_DESMONTAR, t):
        accion, objeto = "desmontar", None
    elif re.search(RE_ARMAR, t):
        accion, objeto = "armar", None
    elif objeto:
        accion = "mostrar"
    elif factor is not None:
        accion = "escalar"
    elif vista:
        accion = "vista"
    elif re.search(r"\breinici|\bresete|\bvolve al inicio|\bcentr|\bdeshace", t):
        accion = "reiniciar"
    elif re.search(r"\b(?:gira|girala|giralo|girar|gire|rota|rotala|rotalo|rotar|rote|voltea|voltealo)\b"
                   r"|\bda(?:le)? (?:una |la )?vuelta", t):
        accion = "rotar"
    else:
        return dict(DESCONOCIDO)

    return {"accion": accion, "objeto": objeto,
            "parametros": {"vista": vista, "factor_escala": factor},
            "confianza": 0.7}  # fija: las reglas no estiman su propia certeza
