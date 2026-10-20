from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import Pedido
from intent import interpretar
from retrieval import buscar_similar, listar_catalogo
from generate3d import generar
from safety import validar_intencion

app = FastAPI(title="TPO Holograma IA")

# El frontend se sirve desde otro puerto en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.post("/intent")
def intent(p: Pedido):
    crudo = interpretar(p.frase)          # llama al LLM, devuelve dict
    return validar_intencion(crudo)       # esquema + filtros


@app.post("/object")
def obtener_objeto(p: Pedido):
    similar = buscar_similar(p.frase, umbral=0.75)
    if similar:
        return {"origen": "catalogo", **similar}
    return {"origen": "generado", **generar(p.frase)}


@app.get("/catalog")
def catalog():
    return listar_catalogo()
