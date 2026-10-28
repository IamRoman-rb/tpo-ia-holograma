import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import generate3d  # noqa: E402
from generate3d import GeneracionFallida, GeneracionNoDisponible, generar  # noqa: E402
from intent import interpretar  # noqa: E402
from retrieval import buscar_similar, catalogo, listar_catalogo, recargar  # noqa: E402
from safety import LimiteFrecuencia, fuera_de_dominio, validar_intencion  # noqa: E402
from schemas import Pedido  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app):
    # Carga el modelo de embeddings al arrancar: si no, el primer pedido tarda varios segundos
    catalogo()
    yield


app = FastAPI(title="TPO Holograma IA", lifespan=lifespan)

# El frontend se sirve desde otro puerto en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "generacion": generate3d.modo()}


@app.post("/intent")
def intent(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    crudo = interpretar(p.frase)          # LLM (o reglas si no hay LLM), devuelve dict
    # La traza (y la respuesta antes de validar) es para el panel de actividad del visor
    traza = {**(crudo.pop("traza", None) or {}), "crudo": {k: v for k, v in crudo.items() if k != "motor"}}
    resultado = validar_intencion(crudo)  # esquema + filtros + límites
    return {**resultado, "traza": traza, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/object")
def obtener_objeto(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    similar = buscar_similar(p.frase)
    if similar:
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
