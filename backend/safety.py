"""Reglas de validación: el LLM propone, las reglas disponen."""
ACCIONES = {"mostrar", "rotar", "escalar", "vista", "reiniciar", "desconocido"}
VISTAS = {"frontal", "trasera", "lateral", "superior", None}


def validar_intencion(d: dict) -> dict:
    if d.get("accion") not in ACCIONES:
        return {"accion": "desconocido", "confianza": 0.0}
    params = d.setdefault("parametros", {})
    if params.get("vista") not in VISTAS:
        params["vista"] = None
    f = params.get("factor_escala")
    if f is not None:
        params["factor_escala"] = max(0.2, min(5.0, float(f)))
    if d.get("confianza", 0) < 0.5:
        d["requiere_confirmacion"] = True
    return d
