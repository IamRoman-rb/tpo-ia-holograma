"""LLM de intención: frase libre -> dict con el esquema de la sección 14.1.A."""
from pathlib import Path

PROMPT_SISTEMA = (Path(__file__).parent / "prompts" / "intent_system.txt").read_text(encoding="utf-8")


def interpretar(frase: str) -> dict:
    # TODO (rol 3): llamar al LLM con PROMPT_SISTEMA, temperatura 0-0.2,
    # parsear el JSON y reintentar si viene inválido.
    return {
        "accion": "desconocido",
        "objeto": None,
        "parametros": {"vista": None, "factor_escala": None},
        "confianza": 0.0,
    }
