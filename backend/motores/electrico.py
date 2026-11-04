"""Motor eléctrico paramétrico, con cada pieza nombrada y su orden y dirección de desarme.

Tres tipos:
  - de inducción trifásico (jaula de ardilla), el de la industria: carcasa con aletas, estator bobinado, rotor de
    jaula, eje, rodamientos, tapas, ventilador, cubreventilador y caja de bornes; con patas o con brida;
  - de corriente continua con escobillas: imanes, inducido, colector y escobillas;
  - brushless de rotor exterior (drones, modelismo): el estator queda adentro y gira la campana con los imanes.
El eje va sobre X. El tamaño sale de la potencia: el radio crece con la raíz cúbica (el volumen, con la potencia).
"""
import math
from dataclasses import dataclass

from geometria import Modelo, caja, cilindro, mover, perfil_engranaje, prisma_anular, rot_x, rot_z, toroide

from .combustion import EJE_X, EJE_Z, _Armado

MATERIALES = {
    "carcasa": ((0.12, 0.32, 0.58), 0.3, 0.5), "hierro": ((0.30, 0.31, 0.33), 0.7, 0.5),
    "acero": ((0.80, 0.80, 0.82), 1.0, 0.25), "cobre": ((0.78, 0.45, 0.20), 0.9, 0.3),
    "aluminio": ((0.72, 0.74, 0.77), 0.9, 0.35), "negro": ((0.07, 0.07, 0.08), 0.2, 0.6),
    "iman": ((0.25, 0.25, 0.30), 0.8, 0.3), "carbon": ((0.15, 0.15, 0.15), 0.1, 0.8),
    "rojo": ((0.65, 0.10, 0.08), 0.3, 0.4),
}


@dataclass(frozen=True)
class Electrico:
    tipo: str = "induccion"        # induccion, continua, brushless
    potencia_kw: float = 4.0
    montaje: str = "patas"         # patas, brida (solo inducción)


def nombre(e: Electrico) -> str:
    base = {"induccion": "Motor eléctrico trifásico", "continua": "Motor de corriente continua",
            "brushless": "Motor brushless"}[e.tipo]
    return base + (" con brida" if e.tipo == "induccion" and e.montaje == "brida" else "")


def generar(e: Electrico) -> Modelo:
    return {"induccion": _induccion, "continua": _continua, "brushless": _brushless}[e.tipo](e)


def _rodamiento(a, x, r_int, r_ext, ancho):
    a.agregar("acero", mover(prisma_anular([(r_ext * math.cos(t), r_ext * math.sin(t))
                                            for t in (2 * math.pi * k / 36 for k in range(36))],
                                           r_ext * 0.82, ancho), (x, 0, 0), EJE_X))
    a.agregar("acero", mover(prisma_anular([(r_int * 1.2 * math.cos(t), r_int * 1.2 * math.sin(t))
                                            for t in (2 * math.pi * k / 36 for k in range(36))],
                                           r_int, ancho), (x, 0, 0), EJE_X))
    for k in range(8):
        t = 2 * math.pi * k / 8
        rb = (r_ext * 0.82 + r_int * 1.2) / 2
        a.agregar("cromo" if "cromo" in a.m.materiales else "acero",
                  mover(cilindro((r_ext * 0.82 - r_int * 1.2) / 2, ancho * 0.6, 10), (x, rb * math.cos(t),
                                                                                       rb * math.sin(t)), EJE_X))


def _induccion(e: Electrico) -> Modelo:
    m = Modelo(MATERIALES)
    a = _Armado(m)
    R = 6.0 * e.potencia_kw ** (1 / 3)             # radio de la carcasa, cm
    L = 2.6 * R                                    # largo de la carcasa
    r_eje = 0.16 * R
    x0, x1 = -L / 2, L / 2

    a.pieza("caja_bornes", "Caja de bornes", "Caja donde se conectan los cables de alimentación.", "caja_bornes",
            (0, 1, 0))
    a.agregar("carcasa", mover(caja(0.7 * R, 0.45 * R, 0.7 * R), (0.1 * L, R + 0.22 * R, 0)))
    a.agregar("negro", mover(cilindro(0.1 * R, 0.25 * R, 12), (0.1 * L, R + 0.2 * R, 0.42 * R), EJE_Z))
    a.pieza("cubreventilador", "Cubreventilador", "Protege el ventilador y dirige el aire sobre las aletas.",
            "cubreventilador", (-1, 0, 0))
    a.agregar("carcasa", mover(prisma_anular([(1.02 * R * math.cos(t), 1.02 * R * math.sin(t))
                                              for t in (2 * math.pi * k / 48 for k in range(48))],
                                             0.97 * R, 0.55 * R), (x0 - 0.45 * R, 0, 0), EJE_X))
    a.agregar("carcasa", mover(cilindro(1.0 * R, 0.04 * R, 48), (x0 - 0.72 * R, 0, 0), EJE_X))
    a.pieza("ventilador", "Ventilador", "Gira con el eje y enfría el motor.", "ventilador", (-1, 0, 0))
    for k in range(8):
        t = 2 * math.pi * k / 8
        a.agregar("negro", mover(mover(caja(0.25 * R, 0.6 * R, 0.04 * R), (0, 0.45 * R, 0)), (x0 - 0.4 * R, 0, 0),
                                 rot_x(t)))
    a.agregar("negro", mover(cilindro(0.25 * R, 0.25 * R, 20), (x0 - 0.4 * R, 0, 0), EJE_X))
    a.pieza("chaveta", "Chaveta", "Pieza que traba la polea o el acople con el eje.", "chaveta", (0, 1, 0))
    a.agregar("acero", mover(caja(0.5 * R, 0.08 * R, 0.08 * R), (x1 + 0.75 * R, r_eje + 0.02 * R, 0)))
    if e.montaje == "brida":
        a.pieza("tapa_delantera", "Brida de montaje", "Tapa delantera con brida para atornillar el motor a la máquina.",
                "tapa_delantera", (1, 0, 0))
        a.agregar("carcasa", mover(cilindro(1.45 * R, 0.12 * R, 48), (x1 + 0.2 * R, 0, 0), EJE_X))
    else:
        a.pieza("tapa_delantera", "Tapa delantera", "Escudo del lado del eje: sostiene el rodamiento delantero.",
                "tapa_delantera", (1, 0, 0))
    a.agregar("carcasa", mover(cilindro(1.0 * R, 0.18 * R, 48), (x1 + 0.09 * R, 0, 0), EJE_X))
    a.pieza("rodamiento_delantero", "Rodamiento delantero", "Sostiene el eje y lo deja girar con poco rozamiento.",
            "rodamiento", (1, 0, 0))
    _rodamiento(a, x1 + 0.09 * R, r_eje, 0.36 * R, 0.14 * R)
    a.pieza("eje", "Eje", "Transmite el giro del rotor a la máquina.", "eje", (1, 0, 0))
    a.agregar("acero", mover(cilindro(r_eje, L + 2.0 * R, 24), (0.3 * R, 0, 0), EJE_X))
    a.pieza("rotor", "Rotor de jaula de ardilla", "Gira por el campo magnético del estator; tiene barras de aluminio.",
            "rotor", (1, 0, 0))
    a.agregar("hierro", mover(cilindro(0.6 * R, 0.8 * L, 40), (0, 0, 0), EJE_X))
    for k in range(18):
        t = 2 * math.pi * k / 18
        a.agregar("aluminio", mover(cilindro(0.04 * R, 0.82 * L, 8), (0, 0.6 * R * math.cos(t), 0.6 * R * math.sin(t)),
                                    EJE_X))
    for x in (-0.41 * L, 0.41 * L):
        a.agregar("aluminio", mover(toroide(0.58 * R, 0.05 * R, 40, 8), (x, 0, 0), EJE_X))
    a.pieza("tapa_trasera", "Tapa trasera", "Escudo del lado del ventilador: sostiene el rodamiento trasero.",
            "tapa_trasera", (-1, 0, 0))
    a.agregar("carcasa", mover(cilindro(1.0 * R, 0.18 * R, 48), (x0 - 0.09 * R, 0, 0), EJE_X))
    a.pieza("rodamiento_trasero", "Rodamiento trasero", "Sostiene el eje del lado del ventilador.", "rodamiento",
            (-1, 0, 0))
    _rodamiento(a, x0 - 0.09 * R, r_eje, 0.32 * R, 0.12 * R)
    a.pieza("estator", "Estator bobinado", "Núcleo de chapas con las bobinas de cobre que crean el campo giratorio.",
            "estator", (1, 0, 0))
    a.agregar("hierro", mover(prisma_anular([(0.86 * R * math.cos(t), 0.86 * R * math.sin(t))
                                             for t in (2 * math.pi * k / 48 for k in range(48))],
                                            0.64 * R, 0.82 * L), (0, 0, 0), EJE_X))
    for x in (-0.45 * L, 0.45 * L):
        a.agregar("cobre", mover(toroide(0.74 * R, 0.09 * R, 40, 10), (x, 0, 0), EJE_X))
    if e.montaje == "patas":
        a.pieza("patas", "Patas de montaje", "Base con agujeros para fijar el motor al piso o a la máquina.", "patas",
                (0, -1, 0))
        for x in (-0.3 * L, 0.3 * L):
            a.agregar("carcasa", mover(caja(0.35 * R, 0.5 * R, 1.6 * R), (x, -R - 0.05 * R, 0)))
        a.agregar("carcasa", mover(caja(0.9 * L, 0.12 * R, 1.8 * R), (0, -R - 0.3 * R, 0)))
    a.pieza("carcasa", "Carcasa con aletas", "Cuerpo exterior: las aletas disipan el calor. Queda fija.", "carcasa",
            fija=True)
    a.agregar("carcasa", mover(prisma_anular([(0.97 * R * math.cos(t), 0.97 * R * math.sin(t))
                                              for t in (2 * math.pi * k / 48 for k in range(48))],
                                             0.87 * R, L), (0, 0, 0), EJE_X))
    for k in range(20):
        t = 2 * math.pi * k / 20
        if abs(math.sin(t) + 1) < 0.15:   # abajo van las patas
            continue
        a.agregar("carcasa", mover(caja(L, 0.18 * R, 0.035 * R), (0, 1.02 * R * math.cos(t), 1.02 * R * math.sin(t)),
                                   rot_x(t)))
    return m


def _continua(e: Electrico) -> Modelo:
    m = Modelo(MATERIALES)
    a = _Armado(m)
    R = 4.0 * e.potencia_kw ** (1 / 3)
    L = 2.2 * R
    r_eje = 0.12 * R
    x0, x1 = -L / 2, L / 2

    for k, z in enumerate((-1, 1)):
        a.pieza(f"escobilla_{k + 1}", f"Escobilla {k + 1}", "Bloque de carbón que lleva la corriente al colector.",
                "escobilla", (0, 0, z))
        a.agregar("carbon", mover(caja(0.25 * R, 0.25 * R, 0.4 * R), (x0 - 0.35 * R, 0, z * 0.42 * R)))
        a.agregar("cobre", mover(cilindro(0.03 * R, 0.4 * R, 8), (x0 - 0.35 * R, 0, z * 0.75 * R), EJE_Z))
    a.pieza("portaescobillas", "Portaescobillas", "Sostiene las escobillas apretadas contra el colector.",
            "portaescobillas", (-1, 0, 0))
    a.agregar("negro", mover(prisma_anular([(0.85 * R * math.cos(t), 0.85 * R * math.sin(t))
                                            for t in (2 * math.pi * k / 36 for k in range(36))],
                                           0.65 * R, 0.12 * R), (x0 - 0.6 * R, 0, 0), EJE_X))
    a.pieza("tapa_trasera", "Tapa trasera", "Cierra el motor del lado del colector.", "tapa_trasera", (-1, 0, 0))
    a.agregar("aluminio", mover(cilindro(1.0 * R, 0.12 * R, 40), (x0 - 0.8 * R, 0, 0), EJE_X))
    a.pieza("tapa_delantera", "Tapa delantera", "Cierra el motor del lado del eje.", "tapa_delantera", (1, 0, 0))
    a.agregar("aluminio", mover(cilindro(1.0 * R, 0.12 * R, 40), (x1 + 0.06 * R, 0, 0), EJE_X))
    for k, x in enumerate((x1 + 0.06 * R, x0 - 0.8 * R)):
        a.pieza(f"rodamiento_{k + 1}", "Rodamiento " + ("delantero" if k == 0 else "trasero"),
                "Sostiene el eje y lo deja girar.", "rodamiento", (1, 0, 0) if k == 0 else (-1, 0, 0))
        _rodamiento(a, x, r_eje, 0.3 * R, 0.1 * R)
    a.pieza("colector", "Colector", "Anillo de delgas de cobre: invierte la corriente en las bobinas al girar.",
            "colector", (-1, 0, 0))
    for k in range(12):
        t = 2 * math.pi * k / 12
        a.agregar("cobre", mover(mover(caja(0.35 * R, 0.08 * R, 0.13 * R), (0, 0.22 * R, 0)), (x0 - 0.35 * R, 0, 0),
                                 rot_x(t)))
    a.pieza("eje", "Eje", "Eje del inducido; sale por adelante.", "eje", (1, 0, 0))
    a.agregar("acero", mover(cilindro(r_eje, L + 2.2 * R, 20), (0.2 * R, 0, 0), EJE_X))
    a.pieza("inducido", "Inducido", "Rotor bobinado: la corriente en sus bobinas y los imanes lo hacen girar.",
            "rotor", (1, 0, 0))
    a.agregar("hierro", mover(prisma_anular(perfil_engranaje(12, 0.45 * R, 0.6 * R), r_eje, 0.8 * L), (0, 0, 0), EJE_X))
    for x in (-0.42 * L, 0.42 * L):
        a.agregar("cobre", mover(toroide(0.48 * R, 0.08 * R, 32, 8), (x, 0, 0), EJE_X))
    for k, t in enumerate((0, math.pi)):
        a.pieza(f"iman_{k + 1}", f"Imán {k + 1}", "Imán permanente que crea el campo magnético fijo.", "iman",
                (0, math.cos(t), math.sin(t)))
        a.agregar("iman", mover(mover(caja(0.75 * L, 0.18 * R, 0.9 * R), (0, 0.75 * R, 0)), R=rot_x(t)))
    a.pieza("carcasa", "Carcasa", "Tubo de acero que cierra el circuito magnético. Queda fija.", "carcasa", fija=True)
    a.agregar("rojo", mover(prisma_anular([(1.0 * R * math.cos(t), 1.0 * R * math.sin(t))
                                           for t in (2 * math.pi * k / 48 for k in range(48))],
                                          0.88 * R, L), (0, 0, 0), EJE_X))
    return m


def _brushless(e: Electrico) -> Modelo:
    m = Modelo(MATERIALES)
    a = _Armado(m)
    R = 2.8 * e.potencia_kw ** (1 / 3) + 1.0
    alto = 0.9 * R
    r_eje = 0.1 * R

    a.pieza("clip", "Anillo de retención", "Traba el eje para que la campana no se salga.", "anillo_retencion",
            (-1, 0, 0))
    a.agregar("acero", mover(toroide(0.14 * R, 0.025 * R, 20, 6), (-0.75 * alto, 0, 0), EJE_X))
    a.pieza("campana", "Campana (rotor)", "Carcasa exterior que gira: lleva los imanes por dentro.", "rotor",
            (1, 0, 0))
    a.agregar("rojo", mover(prisma_anular([(1.0 * R * math.cos(t), 1.0 * R * math.sin(t))
                                           for t in (2 * math.pi * k / 48 for k in range(48))],
                                          0.9 * R, alto), (0.05 * alto, 0, 0), EJE_X))
    a.agregar("rojo", mover(cilindro(1.0 * R, 0.08 * R, 48), (0.55 * alto + 0.04 * R, 0, 0), EJE_X))
    a.pieza("imanes", "Imanes", "Catorce imanes pegados dentro de la campana, con polos alternados.", "iman",
            (1, 0, 0))
    for k in range(14):
        t = 2 * math.pi * k / 14
        a.agregar("iman", mover(mover(caja(0.9 * alto, 0.07 * R, 0.32 * R), (0, 0.85 * R, 0)), (0.05 * alto, 0, 0),
                                rot_x(t)))
    a.pieza("eje", "Eje", "Va fijo a la campana y gira con ella.", "eje", (1, 0, 0))
    a.agregar("acero", mover(cilindro(r_eje, 2.2 * alto, 16), (0.3 * alto, 0, 0), EJE_X))
    a.pieza("rodamientos", "Rodamientos", "Dos rodamientos dentro del estator donde gira el eje.", "rodamiento",
            (1, 0, 0))
    for x in (-0.3 * alto, 0.3 * alto):
        _rodamiento(a, x, r_eje, 0.24 * R, 0.12 * R)
    a.pieza("estator", "Estator bobinado", "Núcleo con 12 polos y sus bobinas de cobre; queda quieto adentro.",
            "estator", (1, 0, 0))
    a.agregar("hierro", mover(prisma_anular(perfil_engranaje(12, 0.55 * R, 0.82 * R), 0.3 * R, 0.85 * alto),
                              (0.05 * alto, 0, 0), EJE_X))
    for k in range(12):
        t = 2 * math.pi * k / 12
        a.agregar("cobre", mover(mover(caja(0.85 * alto, 0.18 * R, 0.16 * R), (0, 0.66 * R, 0)), (0.05 * alto, 0, 0),
                                 rot_x(t)))
    a.pieza("base", "Base de montaje", "Cruz con agujeros para atornillar el motor. Queda fija.", "base", fija=True)
    a.agregar("aluminio", mover(cilindro(0.7 * R, 0.15 * R, 36), (-0.55 * alto, 0, 0), EJE_X))
    for t in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        a.agregar("aluminio", mover(mover(caja(0.12 * R, 1.1 * R, 0.25 * R), (0, 0.6 * R, 0)), (-0.55 * alto, 0, 0),
                                    rot_x(t)))
    return m
