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
import time
from pathlib import Path

import anthropic

import intent_modelo
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


# Si Claude falla por algo que no se arregla solo (sin crédito, clave inválida), no se lo vuelve a llamar por un
# rato: cada intento fallido le suma ~1 s a cada pedido
PAUSA_CLAUDE_S = 600
_claude_pausado_hasta = 0.0


def _motor_suelto(nombre):
    """Función de cada motor propio, importada recién cuando se usa (cada uno carga su modelo)."""
    if nombre == "consenso":
        import consenso
        return lambda f, traza: consenso.interpretar_consenso(f, traza=traza)
    if nombre == "llm_propio":
        import llm_propio
        return lambda f, traza: llm_propio.interpretar_llm_propio(f, traza=traza)
    if nombre == "transformer":
        import intent_transformer
        return lambda f, traza: intent_transformer.interpretar_transformer(f, traza=traza)
    if nombre == "bayes":
        import intent_bayes
        return lambda f, traza: intent_bayes.interpretar_bayes(f, traza=traza)
    if nombre == "modelo":
        return lambda f, traza: intent_modelo.interpretar_modelo(f, traza=traza)
    return lambda f, traza: interpretar_reglas(f)


# Sin Claude: primero el consenso de todos; si no puede (ningún modelo instalado), los motores de a uno
CADENA_PROPIA = ["consenso", "modelo", "reglas"]


def interpretar(frase: str, motor: str | None = None) -> dict:
    """Devuelve la intención (sin validar) e indica qué motor la produjo en "motor".
    Sin motor (o "automatico"): Claude si está disponible; si no, el consenso de los motores propios
    (consenso.py); si no hay ninguno instalado, el intérprete entrenado y al final las reglas.
    Con motor (elegido por voz: "usá Naive Bayes"), ese motor; si falla, las reglas.
    En "traza" queda lo que pasó (modelo, tokens, tiempo o el error): lo muestra el panel de actividad."""
    global _claude_pausado_hasta
    traza = {"modelo": MODELO}
    if motor in (None, "automatico", "claude"):
        if time.monotonic() < _claude_pausado_hasta:
            traza["error"] = "Claude en pausa por un error anterior (sin crédito o sin clave)"
        else:
            try:
                return {**interpretar_llm(frase, traza=traza), "motor": "llm", "traza": traza}
            except (anthropic.AnthropicError, RespuestaInvalida, TypeError) as e:
                # TypeError: el SDK no encontró credenciales.
                log.warning("LLM no disponible (%s: %s); se usan los motores propios", type(e).__name__, e)
                traza["error"] = f"{type(e).__name__}: {e}"[:160]
                if isinstance(e, (TypeError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                                  anthropic.BadRequestError)):
                    _claude_pausado_hasta = time.monotonic() + PAUSA_CLAUDE_S
        cadena = CADENA_PROPIA
    else:
        cadena = [motor, "reglas"]
    for nombre in cadena:
        try:
            traza_motor = {}
            r = _motor_suelto(nombre)(frase, traza_motor)
            return {**r, "motor": nombre, "traza": {**traza, **traza_motor}}
        except Exception as e:   # sin pesos, sin la librería o una respuesta inválida: el siguiente
            log.warning("Motor %s no disponible (%s: %s)", nombre, type(e).__name__, e)
            traza.setdefault("fallas", []).append(f"{nombre}: {type(e).__name__}")
    return {**interpretar_reglas(frase), "motor": "reglas", "traza": traza}


# ------------------------------------------------------------------- motor LLM

def interpretar_llm(frase: str, cliente=None, traza=None) -> dict:
    """traza (opcional): se completa con el modelo que respondió, los tokens, el motivo de fin y el tiempo."""
    cliente = cliente or _cliente_llm()
    t0 = time.perf_counter()
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
    if traza is not None:
        uso = getattr(r, "usage", None)
        traza.update(modelo=getattr(r, "model", None) or MODELO, fin=r.stop_reason,
                     tokens_entrada=getattr(uso, "input_tokens", None), tokens_salida=getattr(uso, "output_tokens", None),
                     ms_llm=round((time.perf_counter() - t0) * 1000))
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

NUMEROS = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}

ESCALAS = [
    (r"\b(\d+(?:[.,]\d+)?)\s*veces\b", None),
    (rf"\b({'|'.join(NUMEROS)})\s+veces\b", None),
    (r"\bel doble\b|\bdoble\b|\bduplica", 2.0),
    (r"\bel triple\b|\btriple\b|\btriplica", 3.0),
    (r"\bla mitad\b|\bmitad\b", 0.5),
    (r"\bagrand|\bmas grande|\baument|\bamplia|\bzoom\b|\bacerca", 1.5),
    (r"\bachic|\bmas chic|\bmas chiqu|\bmas pequen|\breduc|\bdisminu|\baleja", 0.67),
]


def _factor(t):
    for patron, valor in ESCALAS:
        m = re.search(patron, t)
        if m:
            if valor is not None:
                return valor
            n = m.group(1)
            veces = float(NUMEROS[n]) if n in NUMEROS else float(n.replace(",", "."))
            # "tres veces más chico" achica: es 1/3, no 3
            achica = re.search(r"\bmas (?:chic|chiqu|pequen)|\bmenos|\bachic|\breduc", t)
            return round(1 / veces, 3) if achica and veces else veces
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


# Comandos para manejar los modelos ("usá el árbol para los gestos", "¿qué modelo respondió?").
# Cada motor, con las formas de nombrarlo; el orden importa ("la red transformer" es el transformer).
MOTORES = [
    ("transformer", r"\btransformer\b|\broberta\b|\bbert\b|\bdeep learning\b|\baprendizaje profundo\b|\bmodelo profundo\b"),
    ("llm_propio", r"\bllm (?:propio|local|nuestro)\b|\b(?:tu|nuestro|el) (?:propio )?llm\b|\bqwen\b"
                   r"|\bmodelo de lenguaje (?:propio|local)\b"),
    ("claude", r"\bclaude\b|\banthropic\b|\bhaiku\b"),
    ("consenso", r"\bconsenso\b|\bvotacion\b|\bensamble\b|\btodos los (?:modelos|motores)\b|\bcomite\b"),
    ("bayes", r"\bbayes\w*\b|\bmodelo estadistico\b|\bestadistic\w*\b"),
    ("arbol", r"\barbol\w*\b|\brandom forest\b|\bbosque\b"),
    ("modelo", r"\binterprete entrenado\b|\bembeddings?\b|\be5\b|\bclasificador entrenado\b"),
    ("red", r"\bred(?:es)? neuronal\w*\b|\bla red\b|\bmlp\b|\bred profunda\b|\bperceptron\b"),
    ("reglas", r"\breglas\b|\bmotor de reglas\b"),
    ("automatico", r"\bautomatic\w*\b|\bpor defecto\b"),
]
RE_USAR_MOTOR = (r"\b(?:usa|usar|use|usemos|cambia|cambiar|pasa|pasate|activa|activar|elegi|elegir|selecciona|"
                 r"proba|probar|anda|volve|trabaja|interpreta|reconoce|quiero (?:usar|que)|pone|que (?:responda|decida|interprete))\b")
RE_GESTOS = r"\bgest\w*\b|\bmanos?\b|\bdedos?\b"
PREGUNTAS = [
    ("confianza", r"\bque tan segur\w*|\bcuan segur\w*|\bconfianza\b|\bseguridad\b|\bprobabilidad\w*\b|\bsegur\w* estas\b"),
    ("que_motor", r"\b(?:que|cual|quien)\b.*\b(?:modelo|motor|ia|red|interprete|algoritmo)\b"
                  r"|\bquien (?:respondio|contesto|interpreto|lo decidio|decidio)\b|\bcon que (?:lo )?(?:interpret|respond|decid)"),
    ("explicar", r"\bpor que\b|"
                 r"\bexplica\w* (?:por que|como|la (?:decision|respuesta)|el razonamiento|que paso)\b|\bcomo (?:lo )?(?:decidiste|supiste|elegiste|entendiste|interpretaste)\b"
                 r"|\bjustifica\w*\b|\bcomo llegaste\b"),
]
RE_CONSULTAR = (r"\b(?:respondio|contesto|interpreto|decidiste|decidio|entendiste|elegiste|supiste|hiciste|"
                r"segur\w*|confianza|probabilidad|justifica\w*)\b"
                r"|\bexplica\w* (?:por que|como|la (?:decision|respuesta)|el razonamiento|que paso)\b")


def _motor(t):
    return next((nombre for nombre, patron in MOTORES if re.search(patron, t)), None)


def _ambito(t, motor):
    """Para qué se usa el motor: los gestos o los pedidos. El árbol y la red solo existen para los gestos."""
    if motor in ("arbol", "red") or re.search(RE_GESTOS, t):
        return "gestos"
    return "pedidos"


def _pregunta(t):
    return next((nombre for nombre, patron in PREGUNTAS if re.search(patron, t)), None)


def parametros_de_comando(t, accion):
    """Parámetros de usar_motor y consultar (conjuntos cerrados, como la vista): los usan todos los motores."""
    if accion == "usar_motor":
        motor = _motor(t)
        return {"motor": motor, "ambito": _ambito(t, motor) if motor else None}
    if accion == "consultar":
        return {"pregunta": _pregunta(t) or "explicar"}
    return {}


# Desarmado por piezas. El orden importa: "desarmalo" contiene "armalo".
RE_EXPLOTAR = (r"\bexplot|\bdespiece|\bdesarmad[oa]s?\b|\bdesmontad[oa]s?\b"
               r"|\b(?:todas las|las) (?:piezas|partes)\b|\bdesarm\w* (?:todo|entero|completo|del todo)")
RE_DESMONTAR = r"\bdesarm|\bdesmont|\bsac\w* (?:una |la |otra |la siguiente |la proxima )?(?:pieza|parte)|\bsepar"
RE_ARMAR = (r"\b(?:arma|armalo|armala|armar|armalos|monta|montalo|montala|montar|reensambla\w*|ensambla\w*)\b"
            r"|\bvolve a armar|\bjunta (?:las )?piezas")
PALABRAS_DESPIECE = r"\b(?:el |la )?(?:despiece|piezas|partes|desarmad[oa]s?|desmontad[oa]s?|explotad[oa]s?)\b"

# Analizar una pieza sola ("mostrame solo el volante", "aislá el impulsor") y volver al objeto entero
RE_CONJUNTO = (r"\bvolve\w* al (?:conjunto|objeto|modelo)\b|\bsali\w* del analisis|\bmostra\w* todo\b|\bver todo\b"
               r"|\b(?:el )?(?:conjunto|objeto) (?:entero|completo)\b")
RE_AISLAR = r"\baisla\w*|\banaliza\w*|\b(?:solo|solamente|unicamente) (?:el|la|los|las)\b|\bpor separado\b"
_SIN_NOMBRE = {"", "pieza", "esa", "esa pieza", "esta", "esta pieza", "la pieza", "la elegida", "elegida"}


def _pieza(t):
    """Nombre de la pieza a aislar, o None si se refiere a la que ya está elegida ("analizala")."""
    m = re.search(rf"\b(?:solo|solamente|unicamente|aisla\w*|analiza\w*)\s+(?:{ARTICULOS}\s+)?(.*)", t)
    resto = m.group(1) if m else (_objeto(t) or "")
    resto = re.split(r"\b(?:por separado|sola|solo|y|por favor)\b|[.,]", resto)[0]
    resto = re.sub(rf"^(?:{ARTICULOS}\s+)", "", resto.strip()).strip()
    return None if resto in _SIN_NOMBRE else resto


def interpretar_reglas(frase: str) -> dict:
    t = normalizar(frase)
    objeto, vista, factor = _objeto(t), _vista(t), _factor(t)

    if _motor(t) and re.search(RE_USAR_MOTOR, t) and not re.search(RE_CONSULTAR, t):
        return {"accion": "usar_motor", "objeto": None,
                "parametros": {"vista": None, "factor_escala": None, **parametros_de_comando(t, "usar_motor")},
                "confianza": 0.7}
    if re.search(RE_CONSULTAR, t) and _pregunta(t):
        return {"accion": "consultar", "objeto": None,
                "parametros": {"vista": None, "factor_escala": None, **parametros_de_comando(t, "consultar")},
                "confianza": 0.7}
    if re.search(RE_CONJUNTO, t):
        accion, objeto = "conjunto", None
    elif re.search(RE_AISLAR, t):
        accion, objeto = "aislar", _pieza(t)
    elif re.search(RE_EXPLOTAR, t):
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
