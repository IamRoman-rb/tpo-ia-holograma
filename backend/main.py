import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import generate3d  # noqa: E402
from generate3d import GeneracionFallida, GeneracionNoDisponible, generar  # noqa: E402
import intent_modelo  # noqa: E402
from intent import interpretar  # noqa: E402
from retrieval import CARPETA_GLBS, URL_GLBS, buscar_similar, catalogo, listar_catalogo, recargar  # noqa: E402
from safety import LimiteFrecuencia, fuera_de_dominio, validar_intencion  # noqa: E402
from schemas import Pedido  # noqa: E402
import voz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app):
    # Carga los modelos de embeddings al arrancar: si no, el primer pedido tarda varios segundos
    catalogo()
    try:
        intent_modelo.cargar()
    except intent_modelo.ModeloNoDisponible as e:
        logging.getLogger(__name__).warning("Intérprete entrenado no disponible: %s", e)
    # Los motores del consenso (LLM propio, transformer…) y Whisper tardan en cargar: se cargan aparte para no demorar
    # el arranque. Sin esto, el primer pedido tardaba más de un minuto y el visor lo daba por perdido
    threading.Thread(target=_precargar_motores, daemon=True).start()
    if voz.disponible():
        threading.Thread(target=_precargar_voz, daemon=True).start()
    yield


MOTORES = {"listos": False}


def _precargar_motores():
    t0 = time.perf_counter()
    try:
        interpretar("mostrame un engranaje")
    except Exception as e:   # un motor que falta no impide usar los demás
        logging.getLogger(__name__).warning("Precarga de los motores de pedidos: %s", e)
    MOTORES["listos"] = True
    logging.getLogger(__name__).info("Motores de pedidos listos en %.0f s", time.perf_counter() - t0)


def _precargar_voz():
    try:
        voz.cargar()
    except voz.VozNoDisponible as e:
        logging.getLogger(__name__).warning("Voz local no disponible: %s", e)


app = FastAPI(title="TPO Holograma IA", lifespan=lifespan)

# El frontend se sirve desde otro puerto en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Los modelos de Objaverse bajados con scripts/descargar_modelos3d.py (pueden estar en otro disco: HOLOGRAMA_DATOS)
app.mount(URL_GLBS, StaticFiles(directory=CARPETA_GLBS, check_dir=False), name="modelos3d")

# Configurable para mediciones (eval/medir_latencias.py); en uso normal, 30 por minuto
limite = LimiteFrecuencia(maximo=int(os.getenv("LIMITE_PEDIDOS_MINUTO") or 30), ventana_s=60)
# Generar usa GPU (y puede tener costo): límite mucho más bajo (sección 16.5)
limite_generacion = LimiteFrecuencia(maximo=5, ventana_s=600)


def _cliente(request: Request) -> str:
    return request.client.host if request.client else "desconocido"


def _controlar_frecuencia(request: Request):
    if not limite.permitir(_cliente(request)):
        raise HTTPException(429, "Demasiados pedidos seguidos. Esperá un momento.")


@app.get("/health")
def health():
    return {"estado": "ok", "busqueda": catalogo().metodo, "piezas": len(catalogo().items),
            "modelos3d": len(catalogo().objaverse or ()), "motores_listos": MOTORES["listos"],
            "generacion": generate3d.modo(), "voz": f"whisper {voz.MODELO}" if voz.disponible() else None,
            "motores": motores_instalados()}


def motores_instalados():
    """Qué motores propios tienen sus pesos (sin cargarlos): lo muestra la vista del LLM."""
    import consenso
    import intent_bayes
    import intent_transformer
    import llm_propio
    ruta_llm = llm_propio.RUTA_MODELO
    return {
        "llm_propio": ("Qwen 1.5B" if ruta_llm.name.endswith("grande") else "Qwen 0.5B")
        if (ruta_llm / "config.json").exists() else None,
        "transformer": (intent_transformer.RUTA_MODELO / "interprete.json").exists(),
        "modelo": intent_modelo.RUTA_MODELO.exists(),
        "bayes": intent_bayes.RUTA_MODELO.exists(),
        "consenso": "pesos ajustados" if consenso.RUTA_PESOS.exists() else "pesos iguales",
    }


@app.post("/intent")
def intent(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    crudo = interpretar(p.frase, p.motor)  # Claude, el consenso o el motor elegido por voz; devuelve dict
    # La traza (y la respuesta antes de validar) es para el panel de actividad del visor
    traza = {**(crudo.pop("traza", None) or {}), "crudo": {k: v for k, v in crudo.items() if k != "motor"}}
    resultado = validar_intencion(crudo)  # esquema + filtros + límites
    return {**resultado, "traza": traza, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}


MAXIMO_AUDIO = 5 * 1024 * 1024   # ~30 s de opus: alcanza para un pedido


@app.post("/voz")
async def reconocer_voz(request: Request, activacion: bool = False):
    """Audio del micrófono (el cuerpo, tal como lo graba MediaRecorder) -> texto, con Whisper en el equipo.
    Con activacion=true (modo manos libres) se informa si empezó con "holograma" y se devuelve el resto."""
    _controlar_frecuencia(request)
    if not voz.disponible():
        raise HTTPException(503, "El reconocimiento de voz local no está instalado (faster-whisper).")
    audio = await request.body()
    if not audio:
        raise HTTPException(422, "No llegó audio.")
    if len(audio) > MAXIMO_AUDIO:
        raise HTTPException(413, "El audio es demasiado largo.")
    try:
        r = voz.transcribir(audio)
    except voz.VozNoDisponible as e:
        raise HTTPException(503, f"Voz local no disponible: {e}")
    except Exception as e:  # audio que no se puede decodificar
        raise HTTPException(422, f"No se pudo leer el audio ({type(e).__name__}).")
    if activacion:
        r["activado"], r["texto"] = voz.quitar_activacion(r["texto"])
    return r


@app.post("/object")
def obtener_objeto(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    similar = buscar_similar(p.frase)
    if similar:
        if similar["modelo_url"].startswith(URL_GLBS):   # lo sirve el backend, no el frontend
            similar["modelo_url"] = str(request.base_url).rstrip("/") + similar["modelo_url"]
        return {**similar, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}

    # No está en el catálogo: se genera (flujo 2, paso 5)
    if fuera_de_dominio(p.frase):
        raise HTTPException(422, "Ese objeto está fuera del dominio del producto.")
    if not generate3d.disponible():
        raise HTTPException(404, f"No encontré «{p.frase}» en el catálogo y la generación 3D no está activada.")
    if not limite_generacion.permitir(_cliente(request)):
        raise HTTPException(429, "Se alcanzó el límite de modelos generados. Probá en unos minutos.")
    try:
        generado = generar(p.frase)
    except GeneracionNoDisponible:
        raise HTTPException(404, f"No encontré «{p.frase}» en el catálogo y la generación 3D no está activada.")
    except GeneracionFallida as e:
        raise HTTPException(502, f"No se pudo generar el modelo: {e}.")
    recargar()  # para que el próximo pedido parecido lo encuentre en el catálogo
    return {"origen": "generado", **generado, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/catalog")
def catalog():
    return listar_catalogo()
