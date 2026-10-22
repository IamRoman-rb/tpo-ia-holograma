"""Generación 3D: TripoSR / Shap-E / API comercial (se decide con pruebas de calidad, fase 3)."""


class GeneracionNoDisponible(Exception):
    pass


def generar(descripcion: str) -> dict:
    # TODO (rol 3): generar el .glb, guardarlo en el catálogo y devolver
    # {"id", "nombre", "modelo_url"} con origen "generado".
    raise GeneracionNoDisponible("La generación 3D todavía no está integrada")
