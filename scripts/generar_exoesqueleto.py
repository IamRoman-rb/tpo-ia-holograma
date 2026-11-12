"""Genera el exoesqueleto industrial: un traje robótico de diseño propio, por piezas, que se desarma y camina.

Es un exoesqueleto de trabajo (como los de carga en fábricas o depósitos): estructura de tubos de acero, placas de
protección amarillo de seguridad, actuadores hidráulicos a la vista en brazos y piernas, batería en la espalda y casco
con visor horizontal. Cada pieza es un nodo del .glb con nombre, descripción, clase, orden y dirección de desmontaje
(como las autopartes), y las articulaciones traen cómo se mueven (frontend/js/movimiento.js): con "encendé" camina
en el lugar, con los brazos en contrafase de las piernas.

Salidas:
  frontend/models/exoesqueleto/exoesqueleto_industrial.glb
  frontend/models/catalogo.json   se agrega (o se actualiza) el exoesqueleto; lo demás queda igual

Uso (desde la raíz):  python scripts/generar_exoesqueleto.py
Medidas en centímetros (unos 1,87 m de alto).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from geometria import Modelo, a_glb, caja, cilindro, esfera, mover, rot_z, toroide  # noqa: E402

SALIDA = RAIZ / "frontend" / "models" / "exoesqueleto"
CATALOGO = RAIZ / "frontend" / "models" / "catalogo.json"
CATEGORIA = "exoesqueleto"
ID = "exoesqueleto_industrial"
VELOCIDAD = 2 * math.pi * 0.75     # rad/s: un paso de cada pierna cada 1,3 s

EJE_X = rot_z(math.pi / 2)          # del eje Y al eje X (articulaciones que giran de adelante hacia atrás)
FLEXION = (1.0, 0.0, 0.0)           # eje de las articulaciones: la pierna y el brazo van hacia adelante y atrás

MATERIALES = {
    "acero": ((0.62, 0.64, 0.67), 0.95, 0.3), "grafito": ((0.18, 0.19, 0.21), 0.6, 0.45),
    "amarillo": ((0.95, 0.72, 0.08), 0.2, 0.45), "negro": ((0.06, 0.06, 0.07), 0.1, 0.7),
    "aluminio": ((0.78, 0.80, 0.83), 0.9, 0.3), "cromo": ((0.93, 0.93, 0.95), 1.0, 0.08),
    "visor": ((0.08, 0.22, 0.26), 0.7, 0.1), "verde": ((0.20, 0.85, 0.40), 0.0, 0.3),
}

LADOS = {+1: "izquierdo", -1: "derecho"}   # mirando hacia +Z, la izquierda del traje queda en +X


def _alinear(d):
    """Matriz que lleva el eje Y a la dirección d (fórmula de Rodrigues)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(y, d), float(np.dot(y, d))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3, dtype=np.float32) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + k + k @ k / (1 + c)).astype(np.float32)


def tubo(p0, p1, r, seg=20):
    """Cilindro de p0 a p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return mover(cilindro(r, float(np.linalg.norm(p1 - p0)), seg), (p0 + p1) / 2, _alinear(p1 - p0))


def placa(p0, p1, ancho, espesor, frente):
    """Placa de protección a lo largo del segmento p0-p1, corrida hacia adelante (frente, en cm)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    largo = float(np.linalg.norm(p1 - p0))
    centro = (p0 + p1) / 2 + np.array([0.0, 0.0, frente])
    return mover(caja(ancho, 0.8 * largo, espesor), centro, _alinear(p1 - p0))


def actuador(p0, p1, atras):
    """Actuador hidráulico (camisa y vástago) por detrás del segmento p0-p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    corrido = np.array([0.0, 0.0, -atras])
    a, b = p0 + corrido + 0.12 * (p1 - p0), p1 + corrido - 0.12 * (p1 - p0)
    medio = a + 0.55 * (b - a)
    return [("aluminio", tubo(a, medio, 1.3, 14)), ("cromo", tubo(medio, b, 0.7, 12))]


def articulacion(centro, r, ancho):
    return mover(cilindro(r, ancho, 24), centro, EJE_X)


class _Armado:
    def __init__(self, m):
        self.m, self.orden = m, 0

    def pieza(self, id_, nombre, descripcion, clase, direccion=None, fija=False):
        if fija:
            self.m.pieza(id_, nombre, descripcion, clase=clase)
        else:
            self.orden += 1
            self.m.pieza(id_, nombre, descripcion, self.orden, direccion, clase=clase)

    def agregar(self, material, parte):
        self.m.agregar(material, parte)

    def varias(self, partes):
        for material, parte in partes:
            self.m.agregar(material, parte)


def generar() -> Modelo:
    m = Modelo(MATERIALES)
    m.velocidad = VELOCIDAD
    a = _Armado(m)
    hombro = {s: np.array([s * 21.0, 148.0, 0.0]) for s in LADOS}
    codo = {s: np.array([s * 23.0, 120.0, 1.0]) for s in LADOS}
    muneca = {s: np.array([s * 24.0, 93.0, 3.0]) for s in LADOS}
    cadera = {s: np.array([s * 11.0, 98.0, 0.0]) for s in LADOS}
    rodilla = {s: np.array([s * 12.0, 55.0, 2.0]) for s in LADOS}
    tobillo = {s: np.array([s * 12.0, 13.0, 0.0]) for s in LADOS}

    # ---------------------------------------------------------------- 1. casco, batería y peto
    a.pieza("casco", "Casco", "Casco con visor de policarbonato y ventilación; protege la cabeza del operario.",
            "casco", (0, 1, 0))
    a.agregar("amarillo", mover(esfera(12.5), (0, 172, 0)))
    a.agregar("visor", mover(caja(21, 5.5, 4), (0, 171, 10.5)))
    a.agregar("grafito", mover(caja(3, 2.5, 24), (0, 184, 0)))
    for s in LADOS:
        a.agregar("grafito", mover(cilindro(3.2, 2.0, 20), (s * 12.3, 172, 0), EJE_X))
    a.agregar("negro", mover(cilindro(7, 6, 24), (0, 158, 0)))

    a.pieza("bateria", "Batería", "Paquete de baterías de litio que alimenta los actuadores; se cambia en caliente.",
            "bateria", (0, 0, -1))
    a.agregar("grafito", mover(caja(24, 30, 9), (0, 136, -19)))
    for s in LADOS:
        a.agregar("amarillo", mover(cilindro(3.6, 26, 20), (s * 7.5, 136, -25.5)))
    for k in range(4):
        a.agregar("verde", mover(caja(2.2, 1.2, 0.6), (-4.5 + 3 * k, 147, -14.4)))

    a.pieza("peto", "Placa pectoral", "Placa de protección del torso; adentro van los sensores de postura.",
            "placa_pecho", (0, 0, 1))
    a.agregar("amarillo", mover(caja(32, 24, 5), (0, 136, 12)))
    for y in (130, 134, 138):
        a.agregar("negro", mover(caja(24, 1.2, 0.8), (0, y, 14.8)))
    for s in LADOS:
        a.agregar("grafito", mover(caja(4, 26, 3), (s * 14, 136, 8)))

    # ---------------------------------------------------------------- 2. hombreras
    for s, lado in LADOS.items():
        a.pieza(f"hombrera_{lado}", f"Hombrera {lado}", "Cubre el hombro y el motor de la articulación.",
                "hombrera", (s, 0.4, 0))
        a.agregar("amarillo", mover(esfera(8, 24, 12), hombro[s] + np.array([0, 3, 0])))
        a.agregar("grafito", mover(caja(12, 2.5, 14), hombro[s] + np.array([s * 1.5, 9, 0])))

    # ---------------------------------------------------------------- 3. brazos: guantes, antebrazos y brazos
    # Los brazos van en contrafase de la pierna del mismo lado, como al caminar
    for s, lado in LADOS.items():
        fase_brazo = 0.0 if s > 0 else math.pi
        a.pieza(f"guante_{lado}", f"Guante {lado}", "Guante con dedos reforzados y sensores de agarre.",
                "guante", (0, -1, 0))
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=muneca[s], amplitud=0.0, padre=f"antebrazo_{lado}")
        a.agregar("grafito", mover(caja(6, 9, 5), muneca[s] + np.array([0, -6, 0])))
        for k in range(4):
            a.agregar("negro", mover(caja(1.3, 5, 1.4), muneca[s] + np.array([-2.2 + 1.45 * k, -13, 0])))
        a.agregar("negro", mover(caja(1.4, 4, 1.4), muneca[s] + np.array([-s * 3.4, -8, 1.5])))

        a.pieza(f"antebrazo_{lado}", f"Antebrazo {lado}", "Tubo de acero con su actuador: dobla el codo con fuerza.",
                "antebrazo", (s, 0, 0))
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=codo[s], base=-0.4, amplitud=0.15, fase=fase_brazo,
                    padre=f"brazo_{lado}")
        a.agregar("acero", tubo(codo[s], muneca[s], 2.8))
        a.agregar("amarillo", placa(codo[s], muneca[s], 6.5, 3.5, 3.6))
        a.varias(actuador(codo[s], muneca[s], 3.6))
        a.agregar("negro", articulacion(muneca[s], 2.4, 6))

        a.pieza(f"brazo_{lado}", f"Brazo {lado}", "Tubo superior del brazo con el actuador del codo.",
                "brazo", (s, 0, 0))
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=hombro[s], amplitud=0.3, fase=fase_brazo)
        a.agregar("acero", tubo(hombro[s], codo[s], 3.2))
        a.agregar("amarillo", placa(hombro[s], codo[s], 7, 4, 4))
        a.varias(actuador(hombro[s], codo[s], 4.2))
        a.agregar("negro", articulacion(codo[s], 3.6, 8))

    # ---------------------------------------------------------------- 4. piernas: botas, piernas y muslos
    for s, lado in LADOS.items():
        fase_cadera = math.pi if s > 0 else 0.0           # izquierda y derecha en contrafase
        fase_rodilla = math.pi / 2 if s > 0 else -math.pi / 2
        a.pieza(f"bota_{lado}", f"Bota {lado}", "Bota con puntera de acero y sensores de apoyo en la suela.",
                "bota", (0, -1, 0))
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=tobillo[s], amplitud=0.0, padre=f"pierna_{lado}")
        a.agregar("grafito", mover(caja(11, 9, 24), tobillo[s] + np.array([0, -6.5, 4])))
        a.agregar("negro", mover(caja(12, 2, 26), tobillo[s] + np.array([0, -12, 4])))
        a.agregar("amarillo", mover(caja(11.4, 4, 6), tobillo[s] + np.array([0, -8.5, 14])))

        a.pieza(f"pierna_{lado}", f"Pierna {lado}", "Tubo de la pierna con canillera y el actuador de la rodilla.",
                "pierna", (s, 0, 0))
        # la rodilla solo se dobla hacia atrás: base = amplitud, así el ángulo nunca cambia de signo
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=rodilla[s], base=0.3, amplitud=0.3, fase=fase_rodilla,
                    padre=f"muslo_{lado}")
        a.agregar("acero", tubo(rodilla[s], tobillo[s], 3.5))
        a.agregar("amarillo", placa(rodilla[s], tobillo[s], 9, 3.5, 5))
        a.varias(actuador(rodilla[s], tobillo[s], 4.5))
        a.agregar("negro", articulacion(tobillo[s], 3, 8))

        a.pieza(f"muslo_{lado}", f"Muslo {lado}", "Tubo del muslo con el actuador de la cadera y la rodilla.",
                "muslo", (s, 0, 0))
        m.mover_con(tipo="oscilar", eje=FLEXION, centro=cadera[s], amplitud=0.35, fase=fase_cadera)
        a.agregar("acero", tubo(cadera[s], rodilla[s], 4))
        a.agregar("amarillo", placa(cadera[s], rodilla[s], 10, 4, 5))
        a.varias(actuador(cadera[s], rodilla[s], 5))
        a.agregar("negro", articulacion(rodilla[s], 4.5, 10))

    # ---------------------------------------------------------------- 5. cintura y columna
    a.pieza("cinturon", "Cinturón de carga", "Reparte el peso del traje en la cadera; lleva los motores de la cadera.",
            "cinturon", (0, 0, 1))
    a.agregar("grafito", mover(caja(34, 9, 22), (0, 99, 0)))
    a.agregar("amarillo", mover(caja(8, 6, 1.5), (0, 99, 11.5)))
    for s in LADOS:
        a.agregar("negro", articulacion(cadera[s] + np.array([s * 3.5, 0, 0]), 5, 6))

    a.pieza("columna", "Estructura dorsal", "Columna de acero que une hombros y cadera: sostiene todo el traje.",
            "estructura", fija=True)
    a.agregar("acero", tubo((0, 100, -12), (0, 160, -12), 2.5))
    for k in range(6):
        a.agregar("grafito", mover(caja(6, 3, 4), (0, 105 + 10 * k, -12)))
    a.agregar("acero", tubo((-21, 150, -6), (21, 150, -6), 2.0))
    a.agregar("acero", tubo((-12, 100, -8), (12, 100, -8), 2.0))
    a.agregar("grafito", mover(toroide(9, 1.5, 32, 8), (0, 158, -2)))
    return m


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    modelo = generar()
    ruta = SALIDA / f"{ID}.glb"
    ruta.write_bytes(a_glb(modelo, "Exoesqueleto industrial", unidad="cm"))
    fila = {"id": ID, "nombre": "Exoesqueleto industrial",
            "sinonimos": ["exoesqueleto", "exoesqueleto industrial", "traje robótico", "armadura robótica",
                          "armadura mecánica", "traje mecánico", "exoesqueleto de carga", "exosqueleto"],
            "descripcion": "Exoesqueleto de trabajo: casco, batería, placa pectoral, hombreras, brazos, antebrazos, "
                           "guantes, cinturón, muslos, piernas y botas, con actuadores hidráulicos. Camina en el lugar.",
            "url": f"models/exoesqueleto/{ID}.glb", "origen": "catalogo", "categoria": CATEGORIA}
    actual = json.loads(CATALOGO.read_text(encoding="utf-8")) if CATALOGO.exists() else []
    otros = [it for it in actual if it["id"] != ID]
    generados = [it for it in otros if it.get("origen") == "generado"]
    propios = [it for it in otros if it.get("origen") != "generado"]
    with open(CATALOGO, "w", encoding="utf-8") as f:
        json.dump(propios + [fila] + generados, f, ensure_ascii=False, indent=2)
        f.write("\n")
    moviles = sum("movimiento" in p["meta"] for p in modelo.piezas.values())
    print(f"exoesqueleto/{ruta.name} {ruta.stat().st_size / 1024:.1f} KB · {len(modelo.piezas)} piezas "
          f"({moviles} con movimiento) · catalogo.json con {len(propios) + 1 + len(generados)} modelos")


if __name__ == "__main__":
    main()
