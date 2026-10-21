import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from generate3d import GeneracionNoDisponible, generar  # noqa: E402
from intent import interpretar  # noqa: E402
from retrieval import buscar_similar, catalogo, listar_catalogo  # noqa: E402
from safety import LimiteFrecuencia, validar_intencion  # noqa: E402
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

limite = LimiteFrecuencia(maximo=30, ventana_s=60)


def _controlar_frecuencia(request: Request):
    if not limite.permitir(request.client.host if request.client else "desconocido"):
        raise HTTPException(429, "Demasiados pedidos seguidos. Esperá un momento.")


@app.get("/health")
def health():
    return {"estado": "ok", "busqueda": catalogo().metodo, "piezas": len(catalogo().items)}


@app.post("/intent")
def intent(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    crudo = interpretar(p.frase)          # LLM (o reglas si no hay LLM), devuelve dict
    resultado = validar_intencion(crudo)  # esquema + filtros + límites
    return {**resultado, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/object")
def obtener_objeto(p: Pedido, request: Request):
    _controlar_frecuencia(request)
    t0 = time.perf_counter()
    similar = buscar_similar(p.frase)
    if similar:
        return {"origen": "catalogo", **similar, "tiempo_ms": round((time.perf_counter() - t0) * 1000)}
    try:
        return {"origen": "generado", **generar(p.frase), "tiempo_ms": round((time.perf_counter() - t0) * 1000)}
    except GeneracionNoDisponible:
        raise HTTPException(404, f"No encontré «{p.frase}» en el catálogo y la generación 3D todavía no está disponible.")


@app.get("/catalog")
def catalog():
    return listar_catalogo()
