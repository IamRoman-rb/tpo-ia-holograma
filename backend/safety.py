"""Reglas de validación: el LLM propone, las reglas disponen (sección 5.6).

Sistema basado en reglas: determinista, auditable y barato. Toda intención pasa por
acá antes de llegar al visor, venga del LLM o del intérprete por reglas.
"""
import time
from collections import defaultdict, deque

from pydantic import ValidationError

from schemas import MOTORES_GESTOS, MOTORES_PEDIDOS, Intencion
from texto import normalizar

ESCALA_MIN, ESCALA_MAX = 0.2, 5.0
UMBRAL_CONFIRMACION = 0.5

# Objetos fuera del dominio del producto (visor de piezas industriales).
# "bomba" no se incluye a propósito: en el dominio es una bomba de agua o hidráulica.
TERMINOS_BLOQUEADOS = {"arma", "armas", "pistola", "revolver", "fusil", "rifle", "escopeta",
                       "explosivo", "explosivos", "granada", "municion", "municiones"}


def _desconocido(motivo: str) -> dict:
    return {"accion": "desconocido", "objeto": None,
            "parametros": {"vista": None, "factor_escala": None},
            "confianza": 0.0, "motivo": motivo}


def fuera_de_dominio(texto: str) -> bool:
    return bool(set(normalizar(texto).split()) & TERMINOS_BLOQUEADOS)


def validar_intencion(d: dict) -> dict:
    motor = d.get("motor")
    # El consenso decide su propio umbral con la teoría de la decisión (consenso.py); los demás, el de siempre
    umbral = d.get("umbral_confirmacion") or UMBRAL_CONFIRMACION
    d = {k: v for k, v in d.items() if k not in ("motor", "umbral_confirmacion")}

    # 1. Esquema: tipos y valores permitidos
    d.setdefault("parametros", {})
    try:
        intencion = Intencion.model_validate(
            {**d, "confianza": min(1.0, max(0.0, float(d.get("confianza") or 0)))})
    except (ValidationError, TypeError, ValueError):
        return {**_desconocido("respuesta fuera del esquema"), "motor": motor}

    r = intencion.model_dump()

    # 2. Dominio: objetos que el producto no muestra
    if r["objeto"] and fuera_de_dominio(r["objeto"]):
        return {**_desconocido("objeto fuera del dominio del producto"), "motor": motor}

    # 3. Límites
    f = r["parametros"]["factor_escala"]
    if f is not None:
        r["parametros"]["factor_escala"] = max(ESCALA_MIN, min(ESCALA_MAX, f))
    if r["accion"] == "mostrar" and not r["objeto"]:
        return {**_desconocido("falta el objeto a mostrar"), "motor": motor}

    # 4. Comandos de los modelos: un motor que exista para ese ámbito (el árbol no interpreta pedidos)
    p = r["parametros"]
    if r["accion"] == "usar_motor":
        r["objeto"] = None
        validos = MOTORES_GESTOS if p["ambito"] == "gestos" else MOTORES_PEDIDOS
        if p["motor"] is None or p["motor"] not in validos:
            return {**_desconocido("ese motor no existe para " + ("los gestos" if p["ambito"] == "gestos"
                                                                    else "los pedidos")), "motor": motor}
        p["ambito"] = p["ambito"] or "pedidos"
    elif r["accion"] == "consultar":
        r["objeto"] = None
        p["pregunta"] = p["pregunta"] or "explicar"

    # 5. Confianza baja: no se ejecuta sin que el usuario confirme
    r["requiere_confirmacion"] = r["accion"] not in ("desconocido", "consultar") and r["confianza"] < umbral
    r["motor"] = motor
    return r


class LimiteFrecuencia:
    """Límite de pedidos por cliente en una ventana deslizante (sección 16.5: abuso y costos)."""

    def __init__(self, maximo: int = 30, ventana_s: float = 60.0):
        self.maximo, self.ventana_s = maximo, ventana_s
        self._pedidos = defaultdict(deque)

    def permitir(self, cliente: str) -> bool:
        ahora = time.monotonic()
        q = self._pedidos[cliente]
        while q and ahora - q[0] > self.ventana_s:
            q.popleft()
        if len(q) >= self.maximo:
            return False
        q.append(ahora)
        return True
