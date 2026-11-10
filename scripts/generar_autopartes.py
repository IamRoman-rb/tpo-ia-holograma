"""Genera el dataset de autopartes mecánicas: modelos .glb por piezas para manipular y desarmar en el holograma.

Cada autoparte está hecha con las primitivas de backend/geometria.py, igual que el catálogo de demostración: cada
pieza es un nodo del .glb con nombre, descripción, clase, orden y dirección de desmontaje. Así el visor puede sacarlas
de a una, analizarlas y armar la vista explotada.

Salidas:
  frontend/models/autopartes/<id>.glb   un modelo por autoparte
  frontend/models/autopartes/autopartes.json   el dataset: cada autoparte con sus piezas, en orden de desmontaje
  frontend/models/catalogo.json          se agregan (o se actualizan) las autopartes; lo demás queda igual

Uso (desde la raíz):  python scripts/generar_autopartes.py
Las medidas son de escala de visualización (unas 2-3 unidades por modelo), no de fabricación.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from geometria import (Modelo, _parte, caja, circulo, cilindro, cono, escribir_glb, esfera, mover,  # noqa: E402
                       perfil_engranaje, prisma_anular, rot_x, rot_y, rot_z, toroide)

SALIDA = RAIZ / "frontend" / "models" / "autopartes"
CATALOGO = RAIZ / "frontend" / "models" / "catalogo.json"
CATEGORIA = "autoparte"

EJE_X = rot_z(math.pi / 2)    # lleva una pieza del eje Y al eje X
EJE_Z = rot_x(math.pi / 2)    # del eje Y al eje Z (de frente a la cámara)

ARRIBA, ABAJO = (0, 1, 0), (0, -1, 0)
IZQUIERDA, DERECHA = (-1, 0, 0), (1, 0, 0)
ADELANTE, ATRAS = (0, 0, 1), (0, 0, -1)

MATERIALES = {
    "acero": ((0.80, 0.80, 0.82), 1.0, 0.25), "hierro": ((0.30, 0.31, 0.33), 0.7, 0.5),
    "fundicion": ((0.42, 0.40, 0.38), 0.6, 0.7), "aluminio": ((0.72, 0.74, 0.77), 0.9, 0.35),
    "cromo": ((0.93, 0.93, 0.95), 1.0, 0.08), "negro": ((0.07, 0.07, 0.08), 0.2, 0.6),
    "goma": ((0.10, 0.10, 0.10), 0.0, 0.9), "cobre": ((0.72, 0.42, 0.20), 0.9, 0.35),
    "rojo": ((0.62, 0.08, 0.07), 0.3, 0.4), "amarillo": ((0.85, 0.65, 0.10), 0.3, 0.4),
    "friccion": ((0.36, 0.28, 0.22), 0.0, 0.95), "ceramica": ((0.92, 0.91, 0.88), 0.0, 0.3),
    "zinc": ((0.72, 0.74, 0.76), 0.9, 0.3), "escape": ((0.45, 0.35, 0.30), 0.8, 0.6),
    "verde": ((0.20, 0.45, 0.25), 0.3, 0.5), "plastico": ((0.15, 0.15, 0.17), 0.0, 0.5),
}


# ---------------------------------------------------------------- primitivas propias

def helice(radio, y0, y1, vueltas, r_alambre, seg_vuelta=28, seg_tubo=8):
    """Resorte helicoidal sobre el eje Y: un tubo de radio r_alambre que sigue la hélice."""
    total = max(int(vueltas * seg_vuelta), 2)
    p, n, idx = [], [], []
    for k in range(total + 1):
        t = k / total
        a = 2 * math.pi * vueltas * t
        centro = np.array([radio * math.cos(a), y0 + (y1 - y0) * t, radio * math.sin(a)])
        tangente = np.array([-radio * math.sin(a) * 2 * math.pi * vueltas, y1 - y0, radio * math.cos(a) * 2 * math.pi * vueltas])
        tangente /= np.linalg.norm(tangente)
        radial = np.array([math.cos(a), 0.0, math.sin(a)])
        binormal = np.cross(tangente, radial)
        for j in range(seg_tubo):
            b = 2 * math.pi * j / seg_tubo
            d = math.cos(b) * radial + math.sin(b) * binormal
            p.append(centro + r_alambre * d)
            n.append(d)
    for k in range(total):
        for j in range(seg_tubo):
            a0, a1 = k * seg_tubo + j, k * seg_tubo + (j + 1) % seg_tubo
            b0, b1 = a0 + seg_tubo, a1 + seg_tubo
            idx += [(a0, b0, a1), (a1, b0, b1)]
    return _parte(p, n, idx)


def anillo_parcial(r_ext, r_int, h, a0, a1, seg=24):
    """Sector de anillo extruido sobre Y, entre los ángulos a0 y a1 (radianes, en el plano XZ)."""
    p, n, idx = [], [], []
    y0, y1 = -h / 2, h / 2

    def quad(v, normal):
        b = len(p)
        p.extend(v)
        n.extend([normal] * 4)
        idx.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])

    angulos = [a0 + (a1 - a0) * k / seg for k in range(seg + 1)]
    for a, b in zip(angulos, angulos[1:]):
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        cm, sm = math.cos((a + b) / 2), math.sin((a + b) / 2)
        quad([(r_ext * ca, y0, r_ext * sa), (r_ext * cb, y0, r_ext * sb),
              (r_ext * cb, y1, r_ext * sb), (r_ext * ca, y1, r_ext * sa)], (cm, 0, sm))
        quad([(r_int * ca, y0, r_int * sa), (r_int * ca, y1, r_int * sa),
              (r_int * cb, y1, r_int * sb), (r_int * cb, y0, r_int * sb)], (-cm, 0, -sm))
        for y, ny in ((y0, -1), (y1, 1)):
            quad([(r_int * ca, y, r_int * sa), (r_ext * ca, y, r_ext * sa),
                  (r_ext * cb, y, r_ext * sb), (r_int * cb, y, r_int * sb)], (0, ny, 0))
    for a, signo in ((a0, -1), (a1, 1)):
        c, s = math.cos(a), math.sin(a)
        quad([(r_int * c, y0, r_int * s), (r_ext * c, y0, r_ext * s),
              (r_ext * c, y1, r_ext * s), (r_int * c, y1, r_int * s)], (-s * signo, 0, c * signo))
    return _parte(p, n, idx)


def anillo(r_ext, r_int, h, seg=64):
    return prisma_anular(circulo(r_ext, seg), r_int, h)


def alabes(cantidad, r0, r1, ancho, espesor, paso=0.0, curva=0.0):
    """Álabes radiales alrededor del eje Y: placas entre los radios r0 y r1, inclinadas `paso` radianes."""
    partes = []
    for k in range(cantidad):
        a = 2 * math.pi * k / cantidad
        placa = mover(caja(r1 - r0, ancho, espesor), R=rot_x(paso))
        placa = mover(placa, ((r0 + r1) / 2, 0, 0), R=rot_y(curva))
        partes.append(mover(placa, R=rot_y(a)))
    return partes


def radial(parte, radio, angulo, eje="z"):
    """Lleva la parte a `radio` sobre X y la gira `angulo` alrededor del eje (para repetirla en círculo)."""
    giro = {"x": rot_x, "y": rot_y, "z": rot_z}[eje]
    return mover(mover(parte, (radio, 0, 0)), R=giro(angulo))


def en_circulo(cantidad, radio, fase=0.0):
    """Posiciones (x, z) repartidas en un círculo del plano XZ."""
    return [(radio * math.cos(fase + 2 * math.pi * k / cantidad), radio * math.sin(fase + 2 * math.pi * k / cantidad))
            for k in range(cantidad)]


def perfil_leva(r_base, alzada, seg=64):
    """Contorno de una leva: círculo base con una nariz de alzada suave (en el plano XZ)."""
    pts = []
    for k in range(seg):
        a = 2 * math.pi * k / seg
        nariz = alzada * max(0.0, math.cos(a)) ** 3
        pts.append(((r_base + nariz) * math.cos(a), (r_base + nariz) * math.sin(a)))
    return pts


class Armado:
    """Modelo por piezas que lleva la cuenta del orden de desmontaje."""

    def __init__(self):
        self.m, self.orden = Modelo(MATERIALES), 0

    def pieza(self, id_, nombre, descripcion, direccion=None, clase=None):
        """Sin dirección es la pieza base: queda fija al desarmar."""
        if direccion is None:
            self.m.pieza(id_, nombre, descripcion, clase=clase or id_)
        else:
            self.orden += 1
            self.m.pieza(id_, nombre, descripcion, self.orden, direccion, clase=clase or id_)
        return self

    def __call__(self, material, *partes, t=(0, 0, 0), R=None):
        for parte in partes:
            self.m.agregar(material, mover(parte, t, R))
        return self


# ---------------------------------------------------------------- autopartes

def freno_disco():
    a = Armado()
    a.pieza("pinza", "Pinza de freno (caliper)",
            "Abraza el disco; su pistón hidráulico empuja las pastillas cuando se pisa el freno.", ARRIBA)
    a("rojo", caja(0.75, 0.32, 0.13), t=(0, 0.83, 0.21))
    a("rojo", caja(0.75, 0.32, 0.13), t=(0, 0.83, -0.21))
    a("rojo", caja(0.75, 0.12, 0.55), t=(0, 1.04, 0))
    a("rojo", cilindro(0.13, 0.12), t=(0, 0.8, -0.3), R=EJE_Z)
    a("cromo", cilindro(0.035, 0.12, 12), t=(0.2, 1.13, -0.15))
    a.pieza("pastilla_exterior", "Pastilla exterior", "Material de fricción que roza la cara de afuera del disco.",
            ADELANTE, clase="pastilla")
    a("friccion", caja(0.58, 0.24, 0.05), t=(0, 0.8, 0.115))
    a("acero", caja(0.62, 0.27, 0.02), t=(0, 0.8, 0.15))
    a.pieza("pastilla_interior", "Pastilla interior", "Material de fricción que roza la cara de adentro del disco.",
            ATRAS, clase="pastilla")
    a("friccion", caja(0.58, 0.24, 0.05), t=(0, 0.8, -0.115))
    a("acero", caja(0.62, 0.27, 0.02), t=(0, 0.8, -0.15))
    a.pieza("soporte_pinza", "Soporte de la pinza", "Puente atornillado a la mangueta que guía la pinza.", ARRIBA)
    a("fundicion", caja(0.95, 0.12, 0.62), t=(0, 1.16, 0))
    for x in (-0.42, 0.42):
        a("fundicion", caja(0.1, 0.36, 0.1), t=(x, 0.97, -0.26))
    a.pieza("disco", "Disco de freno ventilado",
            "Gira con la rueda; las aletas internas lo ventilan para que no se recaliente.", ADELANTE)
    a("acero", anillo(1.0, 0.46, 0.06, 96), t=(0, 0, 0.065), R=EJE_Z)
    a("acero", anillo(1.0, 0.46, 0.06, 96), t=(0, 0, -0.065), R=EJE_Z)
    a("hierro", *alabes(36, 0.48, 0.98, 0.07, 0.025, curva=0.35), R=EJE_Z)
    a("acero", anillo(0.46, 0.14, 0.05), t=(0, 0, 0.27), R=EJE_Z)
    a("acero", anillo(0.46, 0.42, 0.2), t=(0, 0, 0.16), R=EJE_Z)
    a.pieza("maza", "Maza de rueda con espárragos",
            "Gira sobre su rodamiento y lleva los cinco espárragos donde se ajusta la rueda.")
    a("hierro", cilindro(0.36, 0.3), t=(0, 0, 0.0), R=EJE_Z)
    a("hierro", cilindro(0.2, 0.35), t=(0, 0, -0.3), R=EJE_Z)
    for x, y in en_circulo(5, 0.27, math.pi / 2):
        a("cromo", cilindro(0.03, 0.42, 12), t=(x, y, 0.38), R=EJE_Z)
    return a.m


def amortiguador():
    a = Armado()
    a.pieza("tuerca_superior", "Tuerca del vástago", "Fija el vástago del amortiguador a la copela.", ARRIBA,
            clase="tuerca")
    a("zinc", cilindro(0.08, 0.07, 6, plano=True), t=(0, 1.14, 0))
    a.pieza("copela", "Copela (soporte superior)",
            "Fija el conjunto a la carrocería; tiene un rodamiento para que la rueda pueda girar al doblar.", ARRIBA)
    a("hierro", cilindro(0.3, 0.12), t=(0, 1.03, 0))
    a("goma", cilindro(0.2, 0.08), t=(0, 0.94, 0))
    for x, z in en_circulo(3, 0.24, math.pi / 6):
        a("zinc", cilindro(0.025, 0.2, 10), t=(x, 1.17, z))
    a.pieza("plato_superior", "Plato superior del resorte", "Asiento de arriba del espiral.", ARRIBA,
            clase="plato_resorte")
    a("acero", anillo(0.42, 0.07, 0.04), t=(0, 0.88, 0))
    a.pieza("resorte", "Resorte helicoidal (espiral)",
            "Sostiene el peso del auto; el amortiguador frena sus rebotes.", ARRIBA)
    a("amarillo", helice(0.36, -0.03, 0.85, 5.5, 0.04))
    a.pieza("guardapolvo", "Fuelle guardapolvo", "Protege el vástago del barro y la tierra.", ARRIBA)
    for k in range(7):
        a("goma", toroide(0.1 + 0.015 * (k % 2), 0.025, 32, 8), t=(0, 0.25 + 0.075 * k, 0))
    a.pieza("tope", "Tope de rebote", "Taco de goma que evita el golpe a fondo de la suspensión.", ARRIBA)
    a("goma", cono(0.11, 0.08, 0.14), t=(0, 0.78, 0))
    a.pieza("amortiguador", "Amortiguador con base",
            "Cilindro con aceite que frena el movimiento del resorte. Abajo se abulona a la mangueta.")
    a("negro", cilindro(0.16, 1.4), t=(0, -0.55, 0))
    a("cromo", cilindro(0.045, 0.95), t=(0, 0.6, 0))
    a("acero", anillo(0.42, 0.16, 0.04), t=(0, -0.05, 0))
    a("acero", caja(0.4, 0.42, 0.34), t=(0, -1.35, 0))
    for z in (-0.13, 0.13):
        a("acero", caja(0.4, 0.3, 0.05), t=(0, -1.55, z))
    return a.m


def embrague():
    a = Armado()
    a.pieza("ruleman_empuje", "Rulemán de empuje (crapodina)",
            "Lo mueve el pedal de embrague y aprieta los dedos del diafragma.", ADELANTE)
    a("acero", anillo(0.22, 0.12, 0.12), t=(0, 0, 0.45), R=EJE_Z)
    a("negro", anillo(0.26, 0.2, 0.05), t=(0, 0, 0.53), R=EJE_Z)
    a.pieza("bulones_tapa", "Bulones de la tapa", "Seis bulones que fijan el plato de presión al volante.", ADELANTE,
            clase="bulones")
    for x, y in en_circulo(6, 0.9):
        a("zinc", cilindro(0.04, 0.1, 6, plano=True), t=(x, y, 0.1), R=EJE_Z)
    a.pieza("diafragma", "Tapa y resorte de diafragma",
            "La tapa sujeta el resorte de diafragma: sus dedos hacen de palanca para aflojar el plato.", ADELANTE)
    a("acero", anillo(0.98, 0.82, 0.06), t=(0, 0, 0.03), R=EJE_Z)
    a("acero", anillo(0.84, 0.78, 0.26), t=(0, 0, 0.15), R=EJE_Z)
    a("acero", anillo(0.78, 0.5, 0.03), t=(0, 0, 0.27), R=EJE_Z)
    for k in range(18):
        ang = 2 * math.pi * k / 18
        a("cromo", radial(caja(0.32, 0.07, 0.025), 0.36, ang), t=(0, 0, 0.3))
    a.pieza("plato_presion", "Plato de presión",
            "Aprieta el disco contra el volante: así el motor queda conectado con la caja de cambios.", ADELANTE)
    a("fundicion", anillo(0.8, 0.46, 0.1, 96), t=(0, 0, -0.05), R=EJE_Z)
    a.pieza("disco_embrague", "Disco de embrague",
            "Disco con forro de fricción y resortes amortiguadores; su centro estriado mueve la caja de cambios.",
            ADELANTE)
    a("friccion", anillo(0.78, 0.46, 0.05, 96), t=(0, 0, -0.16), R=EJE_Z)
    a("acero", anillo(0.5, 0.1, 0.02), t=(0, 0, -0.16), R=EJE_Z)
    a("acero", prisma_anular(perfil_engranaje(20, 0.14, 0.17), 0.07, 0.2), t=(0, 0, -0.14), R=EJE_Z)
    for x, y in en_circulo(6, 0.3, 0.3):
        tang = math.atan2(y, x) + math.pi / 2
        a("cromo", mover(cilindro(0.04, 0.16, 12), R=rot_z(tang)), t=(x, y, -0.13))
    a.pieza("volante", "Volante motor",
            "Disco pesado atornillado al cigüeñal; su corona dentada la usa el motor de arranque.")
    a("hierro", anillo(1.0, 0.14, 0.12, 96), t=(0, 0, -0.27), R=EJE_Z)
    a("acero", prisma_anular(perfil_engranaje(90, 1.0, 1.06), 0.96, 0.08), t=(0, 0, -0.27), R=EJE_Z)
    a("hierro", cilindro(0.18, 0.25), t=(0, 0, -0.42), R=EJE_Z)
    return a.m


def turbo():
    a = Armado()
    a.pieza("actuador_wastegate", "Actuador de la wastegate",
            "Cápsula que abre la válvula de descarga cuando la presión de soplado llega al límite.", ATRAS)
    a("acero", cilindro(0.16, 0.12), t=(0.25, -0.05, -0.62), R=EJE_Z)
    a("acero", caja(0.75, 0.03, 0.03), t=(-0.15, -0.05, -0.52))
    a.pieza("carcasa_compresor", "Carcasa del compresor (caracol frío)",
            "Junta el aire comprimido por la rueda y lo manda al motor.", DERECHA)
    a("aluminio", toroide(0.47, 0.22, 64, 16), t=(0.62, 0, 0), R=EJE_X)
    a("aluminio", anillo(0.5, 0.26, 0.3), t=(0.7, 0, 0), R=EJE_X)
    a("aluminio", anillo(0.27, 0.23, 0.45), t=(1.05, 0, 0), R=EJE_X)
    a("aluminio", cilindro(0.13, 0.55), t=(0.62, 0.6, 0.45))
    a.pieza("rueda_compresor", "Rueda del compresor",
            "Aspira aire fresco y lo comprime; gira a más de 100.000 rpm.", DERECHA)
    a("aluminio", cono(0.24, 0.06, 0.32), t=(0.72, 0, 0), R=rot_z(-math.pi / 2))
    a("aluminio", *alabes(12, 0.06, 0.26, 0.28, 0.018, paso=0.5), t=(0.72, 0, 0), R=rot_z(-math.pi / 2))
    a.pieza("carcasa_turbina", "Carcasa de la turbina (caracol caliente)",
            "Recibe los gases de escape y los hace girar contra la rueda de la turbina.", IZQUIERDA)
    a("escape", toroide(0.45, 0.24, 64, 16), t=(-0.62, 0, 0), R=EJE_X)
    a("escape", anillo(0.48, 0.24, 0.3), t=(-0.7, 0, 0), R=EJE_X)
    a("escape", cilindro(0.17, 0.5), t=(-0.62, -0.62, 0.3))
    a("escape", anillo(0.32, 0.18, 0.06), t=(-1.0, 0, 0), R=EJE_X)
    a.pieza("rueda_turbina", "Rueda de la turbina con eje",
            "La mueven los gases de escape; el eje la une a la rueda del compresor.", IZQUIERDA)
    a("hierro", cono(0.22, 0.07, 0.28), t=(-0.72, 0, 0), R=rot_z(math.pi / 2))
    a("hierro", *alabes(11, 0.07, 0.23, 0.24, 0.02, paso=-0.45), t=(-0.72, 0, 0), R=rot_z(math.pi / 2))
    a("cromo", cilindro(0.035, 1.45, 16), R=EJE_X)
    a.pieza("cuerpo_central", "Cuerpo central (CHRA)",
            "Aloja los cojinetes del eje; lo lubrica y enfría el aceite del motor.")
    a("hierro", cilindro(0.26, 0.48), R=EJE_X)
    a("hierro", cilindro(0.08, 0.3), t=(0, 0.3, 0))
    a("hierro", cilindro(0.09, 0.3), t=(0, -0.3, 0))
    return a.m


def alternador():
    a = Armado()
    a.pieza("tuerca_polea", "Tuerca de la polea", "Fija la polea al eje del rotor.", ADELANTE, clase="tuerca")
    a("zinc", cilindro(0.09, 0.07, 6, plano=True), t=(0, 0, 0.9), R=EJE_Z)
    a.pieza("polea", "Polea poli-V", "La mueve la correa del motor.", ADELANTE)
    for k in range(7):
        a("acero", anillo(0.28 if k % 2 else 0.31, 0.06, 0.035), t=(0, 0, 0.66 + 0.035 * k), R=EJE_Z)
    a.pieza("ventilador", "Ventilador", "Refrigera el alternador aspirando aire a través de las tapas.", ADELANTE)
    a("acero", anillo(0.5, 0.06, 0.02), t=(0, 0, 0.55), R=EJE_Z)
    a("acero", *alabes(14, 0.2, 0.52, 0.12, 0.02, curva=0.4), t=(0, 0, 0.5), R=EJE_Z)
    a.pieza("regulador", "Regulador y rectificador",
            "Convierte la corriente alterna en continua y mantiene la tensión en unos 14 V.", ATRAS)
    a("negro", caja(0.65, 0.5, 0.1), t=(0, 0.05, -0.5))
    a("cobre", cilindro(0.05, 0.18, 12), t=(0.2, 0.15, -0.6), R=EJE_Z)
    a.pieza("tapa_trasera", "Tapa trasera", "Carcasa de aluminio con el rodamiento trasero.", ATRAS,
            clase="tapa")
    a("aluminio", anillo(0.63, 0.07, 0.24), t=(0, 0, -0.3), R=EJE_Z)
    a("aluminio", caja(0.18, 0.3, 0.14), t=(0, -0.72, -0.3))
    a.pieza("estator", "Estator", "Núcleo de chapas con los bobinados de cobre donde se genera la corriente.", ATRAS)
    a("hierro", anillo(0.63, 0.46, 0.22), t=(0, 0, 0.05), R=EJE_Z)
    for z in (-0.12, 0.22):
        a("cobre", toroide(0.53, 0.05, 64, 8), t=(0, 0, z), R=EJE_Z)
    a.pieza("rotor", "Rotor de garras con eje",
            "Electroimán que gira dentro del estator: sus garras alternan polos norte y sur.", ATRAS)
    a("hierro", cilindro(0.3, 0.3), t=(0, 0, 0.05), R=EJE_Z)
    for k in range(12):
        ang = 2 * math.pi * k / 12
        z = 0.12 if k % 2 else -0.02
        a("acero", radial(caja(0.08, 0.13, 0.2), 0.37, ang), t=(0, 0, z))
    a("cromo", cilindro(0.05, 1.5, 16), t=(0, 0, 0.1), R=EJE_Z)
    a("cobre", cilindro(0.07, 0.08, 16), t=(0, 0, -0.45), R=EJE_Z)
    a.pieza("tapa_delantera", "Tapa delantera",
            "Carcasa con la oreja de montaje y el rodamiento delantero; se abulona al motor.")
    a("aluminio", anillo(0.63, 0.08, 0.22), t=(0, 0, 0.32), R=EJE_Z)
    a("aluminio", caja(0.18, 0.3, 0.16), t=(0, 0.72, 0.32))
    a("aluminio", cilindro(0.08, 0.2, 16), t=(0, 0.86, 0.32), R=EJE_X)
    return a.m


def motor_arranque():
    a = Armado()
    a.pieza("solenoide", "Automático (solenoide)",
            "Electroimán que empuja el piñón contra la corona y cierra el paso de corriente al motor.", ARRIBA)
    a("acero", cilindro(0.19, 0.75), t=(0.3, 0.52, 0), R=EJE_X)
    for z in (-0.09, 0.09):
        a("cobre", cilindro(0.04, 0.16, 12), t=(-0.15, 0.52, z), R=EJE_X)
    a.pieza("tapa_trasera", "Tapa trasera", "Cierra el motor y sostiene el porta-escobillas.", IZQUIERDA,
            clase="tapa")
    a("aluminio", cilindro(0.33, 0.14), t=(-0.55, 0, 0), R=EJE_X)
    for y, z in ((0.2, 0.2), (-0.2, -0.2)):
        a("zinc", cilindro(0.025, 1.0, 10), t=(-0.05, y, z), R=EJE_X)
    a.pieza("escobillas", "Escobillas", "Carbones que llevan la corriente al colector del inducido.", IZQUIERDA)
    for k in range(4):
        ang = 2 * math.pi * k / 4 + math.pi / 4
        a("negro", caja(0.1, 0.06, 0.06), t=(-0.43, 0.17 * math.cos(ang), 0.17 * math.sin(ang)))
    a.pieza("inducido", "Inducido",
            "Rotor bobinado que gira al recibir corriente; el colector de cobre reparte la corriente.", IZQUIERDA)
    a("hierro", cilindro(0.25, 0.62), t=(0.0, 0, 0), R=EJE_X)
    a("cobre", cilindro(0.15, 0.14), t=(-0.38, 0, 0), R=EJE_X)
    a("cromo", cilindro(0.04, 1.45, 16), t=(0.15, 0, 0), R=EJE_X)
    a.pieza("pinon_bendix", "Piñón (bendix)",
            "Engrana con la corona del volante para hacer girar el motor y se desacopla cuando arranca.", DERECHA)
    a("acero", prisma_anular(perfil_engranaje(10, 0.12, 0.16), 0.04, 0.15), t=(0.95, 0, 0), R=EJE_X)
    a("acero", cilindro(0.1, 0.12), t=(0.82, 0, 0), R=EJE_X)
    a.pieza("carcasa", "Carcasa y nariz", "Yugo con los imanes y nariz con la brida que se abulona a la caja.")
    a("negro", anillo(0.35, 0.29, 0.85), t=(-0.05, 0, 0), R=EJE_X)
    a("aluminio", cono(0.35, 0.22, 0.35), t=(0.55, 0, 0), R=rot_z(-math.pi / 2))
    a("aluminio", caja(0.12, 0.95, 0.5), t=(0.45, 0, 0))
    return a.m


def bomba_agua():
    a = Armado()
    a.pieza("polea", "Polea", "La mueve la correa de distribución o la de accesorios.", ADELANTE)
    a("acero", anillo(0.48, 0.06, 0.05), t=(0, 0, 0.72), R=EJE_Z)
    a("acero", anillo(0.48, 0.42, 0.18), t=(0, 0, 0.64), R=EJE_Z)
    for x, y in en_circulo(4, 0.14):
        a("zinc", cilindro(0.03, 0.08, 6, plano=True), t=(x, y, 0.77), R=EJE_Z)
    a.pieza("impulsor", "Impulsor (turbina)", "Sus paletas empujan el líquido refrigerante por el motor.", ATRAS)
    a("fundicion", anillo(0.42, 0.05, 0.03), t=(0, 0, -0.2), R=EJE_Z)
    a("fundicion", *alabes(8, 0.08, 0.4, 0.14, 0.03, curva=0.6), t=(0, 0, -0.13), R=EJE_Z)
    a.pieza("junta", "Junta", "Sella la bomba contra el block para que no pierda agua.", ATRAS)
    a("goma", anillo(0.62, 0.46, 0.015), t=(0, 0, -0.07), R=EJE_Z)
    a.pieza("eje_rodamiento", "Eje con rodamiento y retén",
            "Eje que une polea e impulsor; el retén evita que el agua llegue al rodamiento.", ADELANTE)
    a("cromo", cilindro(0.05, 1.0, 16), t=(0, 0, 0.3), R=EJE_Z)
    a("acero", cilindro(0.12, 0.35), t=(0, 0, 0.35), R=EJE_Z)
    a("goma", anillo(0.12, 0.05, 0.04), t=(0, 0, 0.02), R=EJE_Z)
    a.pieza("cuerpo", "Cuerpo de la bomba", "Carcasa de aluminio con la brida de montaje y la salida de agua.")
    a("aluminio", anillo(0.62, 0.42, 0.12), t=(0, 0, 0), R=EJE_Z)
    a("aluminio", anillo(0.2, 0.12, 0.5), t=(0, 0, 0.3), R=EJE_Z)
    for x, y in en_circulo(4, 0.72, math.pi / 4):
        a("aluminio", cilindro(0.09, 0.12), t=(x, y, 0), R=EJE_Z)
    a("aluminio", cilindro(0.13, 0.55), t=(0.55, 0.2, 0.05), R=EJE_X)
    return a.m


def piston_biela():
    a = Armado()
    a.pieza("aros", "Aros del pistón",
            "Dos aros de compresión y uno de aceite: sellan la cámara y limpian el aceite del cilindro.", ARRIBA)
    for y, r in ((1.12, 0.022), (1.03, 0.022), (0.93, 0.03)):
        a("cromo", toroide(0.43, r, 64, 8), t=(0, y, 0))
    a.pieza("seguros", "Seguros del perno", "Clips que traban el perno dentro del pistón.", DERECHA)
    for x in (-0.36, 0.36):
        a("acero", toroide(0.09, 0.012, 24, 6), t=(x, 0.78, 0), R=EJE_X)
    a.pieza("perno", "Perno de pistón", "Une el pistón con el pie de la biela y le permite oscilar.", DERECHA)
    a("cromo", cilindro(0.1, 0.72), t=(0, 0.78, 0), R=EJE_X)
    a.pieza("piston", "Pistón", "Recibe la fuerza de la combustión y la transmite a la biela.", ARRIBA)
    a("aluminio", anillo(0.43, 0.38, 0.55), t=(0, 0.9, 0))
    a("aluminio", cilindro(0.43, 0.08), t=(0, 1.19, 0))
    for x in (-0.3, 0.3):
        a("aluminio", cilindro(0.13, 0.12), t=(x, 0.78, 0), R=EJE_X)
    a.pieza("bulones_tapa", "Bulones de la tapa de biela", "Ajustan la tapa contra la biela.", ABAJO,
            clase="bulones")
    for z in (-0.33, 0.33):
        a("zinc", cilindro(0.035, 0.75, 12), t=(0, -0.58, z))
        a("zinc", cilindro(0.06, 0.06, 6, plano=True), t=(0, -0.96, z))
    a.pieza("tapa_biela", "Tapa de biela", "Mitad inferior de la cabeza de biela: abraza el muñón del cigüeñal.", ABAJO)
    a("acero", anillo_parcial(0.42, 0.26, 0.24, math.pi / 2, 3 * math.pi / 2), t=(0, -0.55, 0), R=EJE_X)
    a.pieza("cojinetes", "Cojinetes de biela (metales)",
            "Casquillos antifricción entre la biela y el cigüeñal, lubricados a presión.", ABAJO)
    a("cobre", anillo(0.26, 0.235, 0.22, 48), t=(0, -0.55, 0), R=EJE_X)
    a.pieza("biela", "Biela", "Convierte el movimiento de sube y baja del pistón en giro del cigüeñal.")
    a("acero", anillo(0.2, 0.11, 0.24), t=(0, 0.78, 0), R=EJE_X)
    a("acero", caja(0.22, 1.0, 0.14), t=(0, 0.13, 0))
    a("acero", caja(0.22, 0.14, 0.36), t=(0, -0.38, 0))
    a("acero", anillo_parcial(0.42, 0.26, 0.24, -math.pi / 2, math.pi / 2), t=(0, -0.55, 0), R=EJE_X)
    return a.m


def bujia():
    a = Armado()
    a.pieza("terminal", "Terminal", "Donde se conecta el cable o la bobina.", ARRIBA)
    a("acero", cilindro(0.07, 0.22), t=(0, 1.32, 0))
    a("acero", toroide(0.07, 0.025, 24, 8), t=(0, 1.38, 0))
    a.pieza("junta", "Arandela de sellado", "Sella la bujía contra la tapa de cilindros.", ABAJO)
    a("cobre", toroide(0.24, 0.035, 48, 8), t=(0, -0.2, 0))
    a.pieza("aislador", "Aislador cerámico con electrodo central",
            "Cerámica que aísla la alta tensión; adentro corre el electrodo central (corte didáctico).", ARRIBA)
    a("ceramica", cilindro(0.15, 0.85), t=(0, 0.75, 0))
    for k in range(5):
        a("ceramica", toroide(0.155, 0.025, 32, 8), t=(0, 0.6 + 0.1 * k, 0))
    a("ceramica", cono(0.13, 0.06, 0.45), t=(0, -0.55, 0))
    a("cobre", cilindro(0.025, 2.0, 12), t=(0, 0.2, 0))
    a.pieza("cuerpo", "Cuerpo roscado con electrodo de masa",
            "Hexágono para la llave, rosca y electrodo de masa: la chispa salta entre los dos electrodos.")
    a("zinc", cilindro(0.31, 0.28, 6, plano=True), t=(0, 0.05, 0))
    a("zinc", anillo(0.22, 0.16, 0.18), t=(0, -0.13, 0))
    a("zinc", anillo(0.15, 0.135, 0.55), t=(0, -0.5, 0))
    for k in range(10):
        a("zinc", toroide(0.155, 0.018, 32, 6), t=(0, -0.73 + 0.05 * k, 0))
    a("acero", caja(0.04, 0.17, 0.06), t=(0.12, -0.84, 0))
    a("acero", caja(0.14, 0.04, 0.06), t=(0.07, -0.92, 0))
    return a.m


def rueda():
    a = Armado()
    a.pieza("tapa_central", "Tapa central", "Tapa decorativa que cubre la maza.", ADELANTE)
    a("cromo", cilindro(0.16, 0.05), t=(0, 0, 0.36), R=EJE_Z)
    a.pieza("tuercas", "Tuercas de rueda", "Cinco tuercas que aprietan la llanta contra la maza.", ADELANTE)
    for x, y in en_circulo(5, 0.24, math.pi / 2):
        a("cromo", cilindro(0.05, 0.1, 6, plano=True), t=(x, y, 0.33), R=EJE_Z)
    a.pieza("valvula", "Válvula de inflado", "Por donde se infla el neumático.", ADELANTE)
    a("goma", cilindro(0.03, 0.15, 12), t=(0, -0.8, 0.27), R=EJE_Z)
    a.pieza("neumatico", "Neumático", "Cubierta de goma: da el agarre y absorbe las irregularidades del camino.",
            ADELANTE)
    a("goma", anillo(1.28, 0.86, 0.62, 96), R=EJE_Z)
    for z in (-0.31, 0.31):
        a("goma", toroide(1.08, 0.19, 96, 12), t=(0, 0, z), R=EJE_Z)
    for k in range(48):
        ang = 2 * math.pi * k / 48
        a("goma", radial(caja(0.07, 0.09, 0.58), 1.29, ang))
    a.pieza("llanta", "Llanta", "Rueda de aluminio sobre la que se monta el neumático.", ADELANTE)
    a("aluminio", anillo(0.86, 0.8, 0.66, 96), R=EJE_Z)
    for z in (-0.33, 0.33):
        a("aluminio", anillo(0.92, 0.82, 0.04, 96), t=(0, 0, z), R=EJE_Z)
    a("aluminio", anillo(0.32, 0.18, 0.1), t=(0, 0, 0.22), R=EJE_Z)
    for k in range(5):
        ang = 2 * math.pi * k / 5 + math.pi / 2 + math.pi / 5
        a("aluminio", radial(caja(0.52, 0.12, 0.08), 0.56, ang), t=(0, 0, 0.22))
    a.pieza("maza", "Maza con espárragos", "Gira sobre el rodamiento de rueda; la llanta se ajusta acá.")
    a("hierro", cilindro(0.3, 0.12), t=(0, 0, 0.1), R=EJE_Z)
    a("hierro", cilindro(0.17, 0.4), t=(0, 0, -0.15), R=EJE_Z)
    for x, y in en_circulo(5, 0.24, math.pi / 2):
        a("zinc", cilindro(0.025, 0.3, 10), t=(x, y, 0.22), R=EJE_Z)
    return a.m


def diferencial():
    a = Armado()
    a.pieza("tapa", "Tapa del diferencial", "Cierra la carcasa y contiene el aceite.", DERECHA)
    a("fundicion", cilindro(0.82, 0.06), t=(0.44, 0, 0), R=EJE_X)
    for y, z in en_circulo(10, 0.74):
        a("zinc", cilindro(0.04, 0.06, 6, plano=True), t=(0.49, y, z), R=EJE_X)
    a.pieza("semieje_derecho", "Semieje derecho", "Lleva el giro a la rueda derecha.", DERECHA, clase="semieje")
    a("acero", cilindro(0.07, 1.4, 24), t=(1.05, 0, 0), R=EJE_X)
    a("acero", cilindro(0.25, 0.05), t=(1.77, 0, 0), R=EJE_X)
    a.pieza("semieje_izquierdo", "Semieje izquierdo", "Lleva el giro a la rueda izquierda.", IZQUIERDA,
            clase="semieje")
    a("acero", cilindro(0.07, 1.4, 24), t=(-1.05, 0, 0), R=EJE_X)
    a("acero", cilindro(0.25, 0.05), t=(-1.77, 0, 0), R=EJE_X)
    a.pieza("corona", "Corona", "Engranaje grande que recibe el giro del piñón y reduce la velocidad.", ARRIBA)
    a("acero", prisma_anular(perfil_engranaje(41, 0.6, 0.67), 0.32, 0.12), t=(0.22, 0, 0), R=EJE_X)
    a.pieza("satelites", "Satélites y planetarios",
            "Engranajes que reparten el giro entre las dos ruedas y dejan que giren distinto en las curvas.", ARRIBA)
    for y in (-0.17, 0.17):
        a("acero", prisma_anular(perfil_engranaje(10, 0.1, 0.13), 0.03, 0.08), t=(0, y, 0))
    for x in (-0.17, 0.17):
        a("acero", prisma_anular(perfil_engranaje(14, 0.13, 0.16), 0.07, 0.08), t=(x, 0, 0), R=EJE_X)
    a("cromo", cilindro(0.03, 0.6, 12))
    a.pieza("caja_satelites", "Caja de satélites", "Carcasa que gira con la corona y lleva los satélites.", ARRIBA)
    a("hierro", anillo(0.3, 0.26, 0.2, 48), t=(0.12, 0, 0), R=EJE_X)
    a("hierro", anillo(0.3, 0.1, 0.04), t=(-0.25, 0, 0), R=EJE_X)
    a.pieza("pinon_ataque", "Piñón de ataque",
            "Recibe el giro del cardán y lo transmite a la corona en ángulo recto.", ADELANTE)
    a("acero", prisma_anular(perfil_engranaje(11, 0.16, 0.21), 0.05, 0.16), t=(0.0, 0, 0.58), R=EJE_Z)
    a("acero", cilindro(0.07, 0.9), t=(0.0, 0, 1.05), R=EJE_Z)
    a("acero", cilindro(0.22, 0.05), t=(0.0, 0, 1.52), R=EJE_Z)
    a.pieza("carcasa", "Carcasa (puente)", "Caja de fundición con las trompetas por donde salen los semiejes.")
    a("fundicion", anillo(0.85, 0.78, 0.8, 96), t=(0.0, 0, 0), R=EJE_X)
    a("fundicion", anillo(0.85, 0.12, 0.05, 96), t=(-0.4, 0, 0), R=EJE_X)
    for s in (-1, 1):
        a("fundicion", anillo(0.17, 0.11, 0.9), t=(s * 0.9, 0, 0), R=EJE_X)
    a("fundicion", anillo(0.25, 0.09, 0.5), t=(0.0, 0, 0.95), R=EJE_Z)
    return a.m


def junta_homocinetica():
    a = Armado()
    a.pieza("abrazaderas", "Abrazaderas del fuelle", "Ajustan el fuelle a la campana y al semieje.", DERECHA)
    for x, r in ((0.3, 0.3), (1.0, 0.11)):
        a("zinc", toroide(r, 0.022, 48, 8), t=(x, 0, 0), R=EJE_X)
    a.pieza("fuelle", "Fuelle (guardapolvo)", "Mantiene la grasa adentro y la tierra afuera de la junta.", DERECHA)
    for k in range(8):
        r = 0.29 - 0.024 * k + (0.03 if k % 2 else 0)
        a("goma", toroide(r, 0.035, 48, 8), t=(0.35 + 0.085 * k, 0, 0), R=EJE_X)
    a.pieza("semieje", "Semieje (palier)", "Barra que lleva el giro de la caja a la rueda.", DERECHA)
    a("acero", cilindro(0.07, 1.7, 24), t=(1.0, 0, 0), R=EJE_X)
    a("negro", cilindro(0.16, 0.3), t=(1.95, 0, 0), R=EJE_X)
    a.pieza("nuez", "Nuez (maza interna)", "Estriada al semieje: sus pistas guían las bolas.", DERECHA)
    a("acero", anillo(0.17, 0.07, 0.18, 48), R=EJE_X)
    a.pieza("jaula", "Jaula", "Mantiene las seis bolas en el mismo plano mientras la junta se dobla.", DERECHA)
    a("cromo", anillo(0.29, 0.25, 0.14, 48), R=EJE_X)
    a.pieza("bolas", "Bolas", "Seis bolas que transmiten el giro aunque el semieje esté inclinado.", ARRIBA)
    for y, z in en_circulo(6, 0.23):
        a("cromo", esfera(0.065, 16, 8), t=(0, y, z))
    a.pieza("campana", "Campana con punta de eje",
            "Pista exterior de la junta; la punta estriada entra en la maza de la rueda.")
    a("acero", anillo(0.4, 0.3, 0.32, 48), R=EJE_X)
    a("acero", cono(0.4, 0.2, 0.2), t=(-0.26, 0, 0), R=rot_z(math.pi / 2))
    a("acero", prisma_anular(perfil_engranaje(22, 0.11, 0.13), 0.0001, 0.55), t=(-0.62, 0, 0), R=EJE_X)
    a("acero", cilindro(0.08, 0.2, 24), t=(-0.98, 0, 0), R=EJE_X)
    return a.m


def arbol_levas():
    a = Armado()
    x_cil = (-1.2, -0.4, 0.4, 1.2)
    fases = (0, math.pi, math.pi / 2, 3 * math.pi / 2)   # orden de encendido 1-3-4-2
    a.pieza("polea", "Polea dentada de distribución",
            "La mueve la correa (o cadena) del cigüeñal: el árbol gira a la mitad de vueltas que el motor.", DERECHA)
    a("aluminio", prisma_anular(perfil_engranaje(36, 0.5, 0.55), 0.1, 0.16), t=(1.85, 0, 0), R=EJE_X)
    a("zinc", cilindro(0.08, 0.08, 6, plano=True), t=(1.97, 0, 0), R=EJE_X)
    a.pieza("arbol", "Árbol de levas", "Eje con levas que abren cada válvula en el momento justo.", ARRIBA)
    a("acero", cilindro(0.1, 3.5, 24), t=(0.05, 0, 0), R=EJE_X)
    for x, fase in zip(x_cil, fases):
        for dx in (-0.15, 0.15):
            a("cromo", prisma_anular(perfil_leva(0.2, 0.14), 0.0001, 0.14), t=(x + dx, 0, 0),
              R=rot_x(fase) @ EJE_X)
    for x in (-1.6, -0.8, 0.0, 0.8, 1.6):
        a("acero", cilindro(0.16, 0.12), t=(x, 0, 0), R=EJE_X)
    a.pieza("botadores", "Botadores (taqués)", "Copas que reciben el empuje de la leva y lo pasan a la válvula.",
            ARRIBA)
    for x in x_cil:
        for dx in (-0.15, 0.15):
            a("acero", cilindro(0.11, 0.16), t=(x + dx, -0.32, 0))
    a.pieza("platillos", "Platillos y chavetas", "Retienen el resorte en la punta de cada válvula.", ARRIBA)
    for x in x_cil:
        for dx in (-0.15, 0.15):
            a("acero", anillo(0.12, 0.03, 0.04, 32), t=(x + dx, -0.43, 0))
    a.pieza("resortes", "Resortes de válvula", "Cierran la válvula cuando la leva deja de empujar.", ARRIBA)
    for x in x_cil:
        for dx in (-0.15, 0.15):
            a("cromo", helice(0.09, -0.92, -0.45, 5, 0.017, 20, 6), t=(x + dx, 0, 0))
    a.pieza("valvulas", "Válvulas (admisión y escape)",
            "Abren y cierran el paso de mezcla y de gases de escape en cada cilindro.", ABAJO)
    for x in x_cil:
        for dx in (-0.15, 0.15):
            a("cromo", cilindro(0.025, 1.0, 12), t=(x + dx, -0.92, 0))
            a("escape" if dx > 0 else "acero", cono(0.13, 0.03, 0.08), t=(x + dx, -1.44, 0))
    a.pieza("caballetes", "Caballetes del árbol",
            "Soportes de la tapa de cilindros donde apoya y gira el árbol de levas.")
    for x in (-1.6, -0.8, 0.0, 0.8, 1.6):
        a("aluminio", anillo(0.26, 0.165, 0.14, 32), t=(x, 0, 0), R=EJE_X)
        a("aluminio", caja(0.14, 0.2, 0.6), t=(x, -0.3, 0))
    return a.m


def filtro_aceite():
    a = Armado()
    a.pieza("carcasa", "Carcasa (vaso)", "Lata de acero que contiene el elemento filtrante.", ABAJO)
    a("rojo", anillo(0.42, 0.39, 0.95, 64), t=(0, -0.45, 0))
    a("rojo", cilindro(0.42, 0.04), t=(0, -0.95, 0))
    for k in range(24):
        ang = 2 * math.pi * k / 24
        a("rojo", radial(caja(0.03, 0.12, 0.06), 0.43, ang, "y"), t=(0, -0.85, 0))
    a.pieza("elemento", "Elemento filtrante", "Papel plegado que retiene las partículas del aceite.", ABAJO)
    for k in range(40):
        ang = 2 * math.pi * k / 40
        a("amarillo", radial(caja(0.2, 0.8, 0.015), 0.27, ang, "y"), t=(0, -0.45, 0))
    a("acero", anillo(0.17, 0.14, 0.8, 32), t=(0, -0.45, 0))
    a.pieza("valvula_antidrenaje", "Válvula antidrenaje",
            "Membrana de goma que evita que el aceite vuelva al cárter con el motor apagado.", ABAJO)
    a("goma", anillo(0.3, 0.12, 0.02), t=(0, 0.02, 0))
    a.pieza("junta", "Junta tórica", "Sella el filtro contra el block.", ARRIBA)
    a("goma", toroide(0.34, 0.025, 64, 8), t=(0, 0.11, 0))
    a.pieza("base", "Base roscada", "Placa con la rosca central y los orificios de entrada del aceite.")
    a("acero", anillo(0.41, 0.08, 0.06), t=(0, 0.06, 0))
    a("acero", anillo(0.1, 0.07, 0.12, 24), t=(0, 0.1, 0))
    return a.m


# Cada fila: id, fábrica, nombre, sinónimos (cómo lo puede pedir alguien en Argentina), descripción, sistema del auto
AUTOPARTES = [
    ("freno_disco", freno_disco, "Freno a disco",
     ["freno de disco", "disco de freno", "pinza de freno", "caliper", "frenos del auto", "freno delantero"],
     "Freno a disco con pinza, pastillas, soporte, disco ventilado y maza.", "frenos"),
    ("amortiguador_mcpherson", amortiguador, "Amortiguador McPherson",
     ["amortiguador", "suspensión", "suspension mcpherson", "espiral", "resorte de suspensión", "columna mcpherson"],
     "Columna de suspensión McPherson: amortiguador, espiral, copela, tope y fuelle.", "suspension"),
    ("embrague", embrague, "Kit de embrague",
     ["embrague", "clutch", "disco de embrague", "plato de embrague", "volante motor", "crapodina"],
     "Embrague con volante motor, disco, plato de presión, diafragma y rulemán de empuje.", "transmision"),
    ("turbocompresor", turbo, "Turbocompresor",
     ["turbo", "turbocompresor", "turbina de auto", "turbo del motor", "sobrealimentador"],
     "Turbo con carcasas de turbina y compresor, sus ruedas, eje, cuerpo central y wastegate.", "motor"),
    ("alternador", alternador, "Alternador",
     ["alternador", "generador del auto", "dinamo", "cargador de batería del auto"],
     "Alternador con polea, ventilador, rotor de garras, estator, regulador y tapas.", "electrico"),
    ("motor_arranque", motor_arranque, "Motor de arranque",
     ["motor de arranque", "burro de arranque", "arranque", "burro", "starter"],
     "Motor de arranque con automático, inducido, escobillas, piñón bendix y carcasa.", "electrico"),
    ("bomba_agua", bomba_agua, "Bomba de agua",
     ["bomba de agua", "bomba de refrigerante", "bomba del radiador", "water pump"],
     "Bomba de agua con polea, eje y rodamiento, impulsor, junta y cuerpo.", "refrigeracion"),
    ("piston_biela", piston_biela, "Pistón y biela",
     ["pistón", "biela", "pistón con biela", "conjunto pistón biela", "aros de pistón"],
     "Pistón con aros, perno y seguros, biela, tapa, bulones y cojinetes.", "motor"),
    ("bujia", bujia, "Bujía",
     ["bujía", "bujia de encendido", "chispa", "spark plug"],
     "Bujía de encendido: terminal, aislador cerámico, cuerpo roscado, electrodos y arandela.", "motor"),
    ("rueda", rueda, "Rueda",
     ["rueda", "neumático", "goma del auto", "cubierta", "llanta", "rueda de auto"],
     "Rueda con neumático, llanta de aleación, tuercas, válvula y maza.", "suspension"),
    ("diferencial", diferencial, "Diferencial",
     ["diferencial", "puente trasero", "corona y piñón", "palier trasero", "satélites"],
     "Diferencial con piñón de ataque, corona, satélites, planetarios, semiejes y carcasa.", "transmision"),
    ("junta_homocinetica", junta_homocinetica, "Junta homocinética",
     ["junta homocinética", "homocinética", "punta de eje", "palier", "semieje delantero", "junta cv"],
     "Junta homocinética de bolas con campana, jaula, nuez, bolas, fuelle y semieje.", "transmision"),
    ("arbol_levas", arbol_levas, "Árbol de levas y válvulas",
     ["árbol de levas", "levas", "distribución", "válvulas del motor", "tren de válvulas", "botadores"],
     "Árbol de levas de 4 cilindros con polea, botadores, resortes y 8 válvulas.", "motor"),
    ("filtro_aceite", filtro_aceite, "Filtro de aceite",
     ["filtro de aceite", "filtro del motor", "oil filter"],
     "Filtro de aceite con carcasa, elemento de papel, válvula antidrenaje, junta y base.", "motor"),
]


def piezas_del_modelo(modelo):
    """Las piezas en el orden en que se desmontan (las fijas al final), como las lee el visor."""
    metas = [p["meta"] for p in modelo.piezas.values()]
    return sorted(metas, key=lambda m: (m["orden"] is None, m["orden"] or 0))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    dataset, filas = [], []
    for id_, fabricar, nombre, sinonimos, descripcion, sistema in AUTOPARTES:
        modelo = fabricar()
        ruta = SALIDA / f"{id_}.glb"
        escribir_glb(modelo, ruta, id_)
        url = f"models/autopartes/{id_}.glb"
        piezas = piezas_del_modelo(modelo)
        filas.append({"id": id_, "nombre": nombre, "sinonimos": sinonimos, "descripcion": descripcion,
                      "url": url, "origen": "catalogo", "categoria": CATEGORIA})
        dataset.append({"id": id_, "nombre": nombre, "sistema": sistema, "url": url, "piezas": [
            {"pieza": m["pieza"], "nombre": m["nombre"], "clase": m["clase"], "descripcion": m["descripcion"],
             "orden": m["orden"], "direccion": m["direccion"]} for m in piezas]})
        print(f"autopartes/{ruta.name:28s} {ruta.stat().st_size / 1024:7.1f} KB · {len(piezas):2d} piezas")

    with open(SALIDA / "autopartes.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Alta en el catálogo: se reemplazan las autopartes de una corrida anterior y se conserva todo lo demás
    actual = json.loads(CATALOGO.read_text(encoding="utf-8")) if CATALOGO.exists() else []
    ids = {x["id"] for x in filas}
    otros = [it for it in actual if it.get("categoria") != CATEGORIA and it["id"] not in ids]
    generados = [it for it in otros if it.get("origen") == "generado"]
    propios = [it for it in otros if it.get("origen") != "generado"]
    with open(CATALOGO, "w", encoding="utf-8") as f:
        json.dump(propios + filas + generados, f, ensure_ascii=False, indent=2)
        f.write("\n")
    total = sum(len(d["piezas"]) for d in dataset)
    print(f"{len(dataset)} autopartes, {total} piezas · catalogo.json con {len(propios) + len(filas) + len(generados)} modelos")


if __name__ == "__main__":
    main()
