"""Generación 3D (secciones 5.4 y 7.3, flujo 2, paso 5).

Si el objeto pedido no está en el catálogo, se genera, se guarda y se agrega al catálogo:
el próximo pedido parecido se resuelve al instante por búsqueda (sección 4: "el sistema
reutiliza modelos generados antes") y conserva la etiqueta de "generado por IA".

Dos formas de generar (GEN3D_MODO en .env):
    ensamble   El LLM diseña el objeto POR PIEZAS (backend/ensamble.py): se puede desarmar.
               Solo necesita la clave del LLM; no usa GPU.
    remoto     TripoSR o Shap-E en Colab (backend/generador_colab/servidor_generacion.ipynb):
               una sola malla, que no se puede desarmar. Necesita GEN3D_URL y GEN3D_TOKEN.
    auto       (por defecto) ensamble si hay clave del LLM; si no, remoto si hay GEN3D_URL.
    desactivado
"""
import json
import logging
import os
import re
import struct
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx

import ensamble
from intent import MODELO, MODELOS_CON_FALLBACK

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CATALOGO = RAIZ / "frontend" / "models" / "catalogo.json"
DIR_GENERADOS = RAIZ / "frontend" / "models" / "generados"
TAMANO_MAXIMO = 20 * 1024 * 1024  # sección 16.5: validar tamaño y formato antes de cargar
ESPERA_S = 300                     # la generación tarda de decenas de segundos a unos minutos
PROMPT_B = (Path(__file__).parent / "prompts" / "gen3d_prompt.txt").read_text(encoding="utf-8")
ESTILO = ("single object, centered, plain white background, three-quarter view, no text, no people, "
          "realistic style, clear materials")

_candado = threading.Lock()  # una generación a la vez: la GPU es una sola


class GeneracionNoDisponible(Exception):
    pass


class GeneracionFallida(Exception):
    pass


# ------------------------------------------------------------------ utilidades

def identificador(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return "gen_" + (re.sub(r"[^a-z0-9]+", "_", t).strip("_")[:60] or "objeto")


def es_glb_valido(datos: bytes) -> bool:
    """Cabecera glTF binaria: 'glTF', versión 2 y largo declarado igual al real."""
    if not 20 <= len(datos) <= TAMANO_MAXIMO:
        return False
    magia, version, largo = struct.unpack_from("<4sII", datos)
    return magia == b"glTF" and version == 2 and largo == len(datos)


def prompt_en_ingles(descripcion: str) -> str:
    """Prompt B (sección 14.1): lo escribe el LLM. Sin LLM, se agrega el estilo al texto original."""
    try:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}             if MODELO in MODELOS_CON_FALLBACK else {}
        r = anthropic.Anthropic(timeout=15.0, max_retries=1).beta.messages.create(
            model=MODELO,
            max_tokens=1024,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROMPT_B.replace("{objeto}", descripcion)}],
            **extra,
        )
        texto = next((b.text for b in r.content if b.type == "text"), "").strip().strip('"')
        if r.stop_reason == "end_turn" and texto:
            return texto
    except (anthropic.AnthropicError, TypeError) as e:
        log.info("Prompt B sin LLM (%s); se usa la descripción original", type(e).__name__)
    return f"{descripcion}, {ESTILO}"


# ------------------------------------------------------------------ servidor de generación

_cliente_http = None


def _http() -> httpx.Client:
    global _cliente_http
    if _cliente_http is None:
        _cliente_http = httpx.Client(timeout=ESPERA_S)
    return _cliente_http


def _generar_remoto(prompt: str) -> bytes:
    url = os.getenv("GEN3D_URL", "").rstrip("/")
    try:
        r = _http().post(f"{url}/generar", json={"prompt": prompt},
                         headers={"Authorization": f"Bearer {os.getenv('GEN3D_TOKEN', '')}"})
    except httpx.HTTPError as e:
        raise GeneracionFallida(f"el servidor de generación no respondió ({type(e).__name__})") from e
    if r.status_code != 200:
        raise GeneracionFallida(f"el servidor de generación respondió {r.status_code}")
    return r.content


# ------------------------------------------------------------------ catálogo

def registrar_en_catalogo(id_: str, descripcion: str, url: str, prompt: str, segundos: float,
                          nombre: str | None = None, metodo: str = "remoto", piezas: int = 1) -> dict:
    items = json.loads(RUTA_CATALOGO.read_text(encoding="utf-8"))
    existente = next((it for it in items if it["id"] == id_), None)
    if existente:
        return existente
    item = {
        "id": id_,
        "nombre": nombre or descripcion[:1].upper() + descripcion[1:],
        "sinonimos": list(dict.fromkeys([descripcion] + ([nombre.lower()] if nombre else []))),
        "descripcion": f"Modelo generado por IA a partir de «{descripcion}». No apto para fabricación sin revisión.",
        "url": url,
        "origen": "generado",
        "metodo": metodo,
        "piezas": piezas,
        "prompt": prompt,
        "segundos_generacion": round(segundos, 1),
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    items.append(item)
    RUTA_CATALOGO.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return item


# ------------------------------------------------------------------ punto de entrada

def _hay_credenciales_llm() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def modo() -> str:
    """Método de generación activo: "ensamble", "remoto" o "desactivado"."""
    pedido = os.getenv("GEN3D_MODO", "auto")
    if pedido == "ensamble":
        return "ensamble" if _hay_credenciales_llm() else "desactivado"
    if pedido == "remoto":
        return "remoto" if os.getenv("GEN3D_URL") else "desactivado"
    if pedido == "auto":
        if _hay_credenciales_llm():
            return "ensamble"
        return "remoto" if os.getenv("GEN3D_URL") else "desactivado"
    return "desactivado"


def disponible() -> bool:
    return modo() != "desactivado"


def _generar_por_piezas(descripcion: str):
    try:
        return ensamble.generar_glb(descripcion)
    except ensamble.EnsambleInvalido as e:
        raise GeneracionFallida(str(e)) from e
    except (anthropic.AnthropicError, TypeError) as e:
        raise GeneracionFallida(f"el LLM no respondió ({type(e).__name__})") from e


def generar(descripcion: str) -> dict:
    if not disponible():
        raise GeneracionNoDisponible("La generación 3D no está configurada")
    id_ = identificador(descripcion)
    url = f"models/generados/{id_}.glb"
    ruta = DIR_GENERADOS / f"{id_}.glb"
    with _candado:
        segundos, prompt, nombre, metodo, cantidad = 0.0, "", None, modo(), 1
        if not ruta.exists():  # puede haberse generado mientras esperaba el candado
            t0 = time.perf_counter()
            if metodo == "ensamble":
                datos, diseno = _generar_por_piezas(descripcion)
                nombre, cantidad = diseno.nombre, len(diseno.piezas)
            else:
                prompt = prompt_en_ingles(descripcion)
                datos = _generar_remoto(prompt)
            if not es_glb_valido(datos):
                raise GeneracionFallida("el archivo generado no es un .glb válido")
            DIR_GENERADOS.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(datos)
            segundos = time.perf_counter() - t0
            log.info("Generado %s en %.1f s", id_, segundos)
        item = registrar_en_catalogo(id_, descripcion, url, prompt, segundos, nombre, metodo, cantidad)
    return {"id": item["id"], "nombre": item["nombre"], "modelo_url": item["url"],
            "segundos_generacion": item.get("segundos_generacion"), "piezas": item.get("piezas", 1)}
