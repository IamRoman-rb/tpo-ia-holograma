"""Motor de combustión paramétrico, con cada pieza nombrada y su orden y dirección de desarme.

Familias: en línea (1 a 6 cilindros), en V, bóxer (cilindros opuestos) y radial (aviación). El cigüeñal va sobre el
eje X; cada banco de cilindros apunta en una dirección u del plano YZ (en línea: arriba; en V: ±la mitad del ángulo;
bóxer: a los costados). Las piezas de un banco se arman como si fuera un motor en línea vertical (x a lo largo del
cigüeñal, y hacia arriba del cilindro, z a un costado) y después se giran a su dirección.

Las medidas están en centímetros (diámetro del cilindro, carrera…) y el resto se deriva de ellas, así un mismo pedido
siempre da el mismo motor. Cada pieza lleva su clase ("piston" para "Pistón 3"): son las etiquetas con las que se
entrena el reconocimiento de piezas (ml/piezas/).
"""
import math
from dataclasses import dataclass

import numpy as np

from geometria import (Modelo, caja, cilindro, cono, mover, perfil_engranaje, prisma_anular, rot_x, rot_z,
                       toroide)

EJE_X = rot_z(math.pi / 2)    # lleva un cilindro del eje Y al eje X (el del cigüeñal)
EJE_Z = rot_x(math.pi / 2)    # del eje Y al eje Z

# Velocidad del cigüeñal en la animación (rad/s): una vuelta por segundo se sigue bien con la vista
VELOCIDAD = 2 * math.pi

MATERIALES = {
    "aluminio": ((0.72, 0.74, 0.77), 0.9, 0.35), "hierro": ((0.30, 0.31, 0.33), 0.7, 0.5),
    "acero": ((0.80, 0.80, 0.82), 1.0, 0.25), "cromo": ((0.93, 0.93, 0.95), 1.0, 0.08),
    "negro": ((0.07, 0.07, 0.08), 0.2, 0.6), "rojo": ((0.62, 0.08, 0.07), 0.3, 0.4),
    "cobre": ((0.72, 0.42, 0.20), 0.9, 0.35), "goma": ((0.10, 0.10, 0.10), 0.0, 0.9),
    "ceramica": ((0.92, 0.91, 0.88), 0.0, 0.3), "escape": ((0.45, 0.35, 0.30), 0.8, 0.6),
    "madera": ((0.55, 0.36, 0.18), 0.0, 0.7),
}


@dataclass(frozen=True)
class Combustion:
    cilindros: int = 4
    disposicion: str = "linea"     # linea, v, boxer, radial
    angulo_v: float = 90.0         # grados entre bancos (motores en V)
    diametro: float = 8.6          # cm
    carrera: float = 8.6           # cm
    arboles: str = "dohc"          # sohc, dohc (un árbol de levas por banco o dos)
    valvulas: int = 4              # por cilindro: 2 o 4
    turbo: bool = False
    distribucion: str = "correa"   # correa, cadena
    combustible: str = "nafta"     # nafta, diesel


def nombre(c: Combustion) -> str:
    base = {"linea": f"Motor de {c.cilindros} cilindros en línea" if c.cilindros > 1 else "Motor monocilíndrico",
            "v": f"Motor V{c.cilindros}", "boxer": f"Motor bóxer de {c.cilindros} cilindros",
            "radial": f"Motor radial de {c.cilindros} cilindros"}[c.disposicion]
    return base + (" turbo" if c.turbo else "") + (" diésel" if c.combustible == "diesel" else "")


def _hacia(fi):
    """Matriz que lleva el sistema del banco (y = eje del cilindro) a la dirección fi (radianes desde arriba)."""
    return rot_x(fi)


def _dir(fi):
    return (0.0, math.cos(fi), math.sin(fi))


class _Armado:
    """Lleva la cuenta del orden de desarme: cada pieza que se agrega sale después de la anterior."""

    def __init__(self, m):
        self.m, self.orden = m, 0

    def pieza(self, id_, nombre_, descripcion, clase, direccion=None, fija=False):
        if fija:
            self.m.pieza(id_, nombre_, descripcion, clase=clase)
        else:
            self.orden += 1
            self.m.pieza(id_, nombre_, descripcion, self.orden, direccion, clase=clase)

    def mover(self, **movimiento):
        self.m.mover_con(**movimiento)

    def agregar(self, material, parte, R=None):
        self.m.agregar(material, mover(parte, R=R) if R is not None else parte)


def generar(c: Combustion) -> Modelo:
    if c.disposicion == "radial":
        return _radial(c)
    m = Modelo(MATERIALES)
    m.velocidad = VELOCIDAD
    a = _Armado(m)
    D, r = c.diametro, c.carrera / 2
    L = 1.75 * c.carrera                       # largo de la biela
    paso = 1.3 * D                             # distancia entre cilindros
    if c.disposicion == "linea":
        bancos = [0.0]
    elif c.disposicion == "v":
        bancos = [-math.radians(c.angulo_v) / 2, math.radians(c.angulo_v) / 2]
    else:
        bancos = [-math.pi / 2, math.pi / 2]
    n_banco = max(1, c.cilindros // len(bancos))
    largo = n_banco * paso + 0.5 * D
    cubierta = r + L + 0.6 * D                 # altura de la cara superior del bloque, desde el eje del cigüeñal
    alto_culata = 0.9 * D
    tope_culata = cubierta + 0.12 + alto_culata
    desfase = [0.0] if len(bancos) == 1 else [-0.22 * D, 0.22 * D]   # las bielas de un par comparten muñón
    xs = [(k - (n_banco - 1) / 2) * paso for k in range(n_banco)]
    # ángulo de cada muñón del cigüeñal (alternados: dos pistones arriba, dos abajo…)
    giro = [math.pi * ((k + (k >= n_banco / 2)) % 2) for k in range(n_banco)]
    ancho_total = max(abs(cubierta * math.sin(fi)) for fi in bancos) + 0.8 * D
    alto_total = max(cubierta * math.cos(fi) for fi in bancos)

    def lado(b):
        """Nombre del banco para piezas repetidas en motores en V o bóxer."""
        return "" if len(bancos) == 1 else (" izquierdo" if bancos[b] < 0 else " derecho")

    def numero(b, k):
        return b * n_banco + k + 1

    # Del lado de adentro de la V va la admisión; del de afuera, el escape (en línea: admisión a la derecha)
    lado_admision = [1 if fi <= 0 else -1 for fi in bancos]

    # ---------------------------------------------------------------- 1. accesorios y lo que va por fuera
    for b, fi in enumerate(bancos):
        R, u, s = _hacia(fi), _dir(fi), lado_admision[b]
        a.pieza(f"multiple_escape_{b}", f"Múltiple de escape{lado(b)}",
                "Junta los gases que salen de cada cilindro y los lleva al escape (o al turbo).", "multiple_escape",
                _dir(fi - s * math.pi / 2) if len(bancos) > 1 else (0, 0, -s))
        for x in xs:
            a.agregar("escape", mover(cilindro(0.15 * D, 0.9 * D, 20), (x + desfase[b], cubierta + 0.45 * D,
                                                                       -s * (0.75 * D)), EJE_Z), R)
        a.agregar("escape", mover(cilindro(0.24 * D, largo, 24), (desfase[b], cubierta + 0.2 * D, -s * 1.25 * D),
                                  EJE_X), R)
        if c.turbo:
            a.pieza(f"turbo_{b}", "Turbo" + lado(b),
                    "Turbocompresor: los gases de escape hacen girar una turbina que comprime el aire de admisión.",
                    "turbo", (1, 0, 0))
            x0 = largo / 2 + 0.6 * D
            centro = (x0, cubierta + 0.2 * D, -s * 1.25 * D)
            a.agregar("escape", mover(toroide(0.42 * D, 0.2 * D, 32, 12), centro, EJE_X), R)
            a.agregar("aluminio", mover(toroide(0.36 * D, 0.17 * D, 32, 12), (x0 + 0.75 * D, *centro[1:]), EJE_X), R)
            a.agregar("hierro", mover(cilindro(0.18 * D, 0.75 * D, 20), (x0 + 0.37 * D, *centro[1:]), EJE_X), R)
        a.pieza(f"multiple_admision_{b}", f"Múltiple de admisión{lado(b)}",
                "Reparte el aire (y en los de inyección indirecta, la mezcla) entre los cilindros.", "multiple_admision",
                _dir(fi + s * math.pi / 2) if len(bancos) > 1 else (0, 0, s))
        for x in xs:
            a.agregar("aluminio", mover(cilindro(0.17 * D, 0.8 * D, 20), (x + desfase[b], cubierta + 0.5 * D,
                                                                         s * 0.75 * D), EJE_Z), R)
        a.agregar("aluminio", mover(cilindro(0.34 * D, largo, 24), (desfase[b], cubierta + 0.5 * D, s * 1.25 * D),
                                    EJE_X), R)

    a.pieza("alternador", "Alternador", "Genera la electricidad del auto y carga la batería; lo mueve una correa.",
            "alternador", (0, 0, -1))
    centro = (largo / 2 - 0.2 * D, -0.1 * D, -(ancho_total + 0.55 * D))
    a.agregar("aluminio", mover(cilindro(0.5 * D, 0.9 * D, 28), centro, EJE_X))
    a.agregar("acero", mover(cilindro(0.3 * D, 0.15 * D, 24), (centro[0] + 0.55 * D, *centro[1:]), EJE_X))
    a.pieza("motor_arranque", "Motor de arranque", "Motor eléctrico que hace girar el volante para arrancar.",
            "motor_arranque", (0, 0, 1))
    centro = (-largo / 2 + 0.3 * D, -r - 0.1 * D, ancho_total + 0.45 * D)
    a.agregar("negro", mover(cilindro(0.36 * D, 1.2 * D, 24), centro, EJE_X))
    a.agregar("acero", mover(cilindro(0.12 * D, 0.4 * D, 16), (centro[0] - 0.75 * D, *centro[1:]), EJE_X))
    a.pieza("filtro_aceite", "Filtro de aceite", "Retiene las partículas del aceite que lubrica el motor.",
            "filtro_aceite", (0, 0, 1))
    a.agregar("rojo", mover(cilindro(0.4 * D, 0.9 * D, 28), (0.15 * largo, -0.2 * D, ancho_total + 0.4 * D), EJE_Z))
    a.pieza("bomba_agua", "Bomba de agua", "Hace circular el refrigerante entre el motor y el radiador.",
            "bomba_agua", (1, 0, 0))
    centro = (largo / 2 + 0.25 * D, 0.55 * alto_total, 0)
    a.agregar("aluminio", mover(cilindro(0.38 * D, 0.5 * D, 24), centro, EJE_X))
    a.agregar("acero", mover(cilindro(0.5 * D, 0.08 * D, 28), (centro[0] + 0.3 * D, *centro[1:]), EJE_X))

    # ---------------------------------------------------------------- 2. arriba: bujías, tapas, distribución, levas
    for b, fi in enumerate(bancos):
        R, u = _hacia(fi), _dir(fi)
        for k, x in enumerate(xs):
            n = numero(b, k)
            if c.combustible == "diesel":
                a.pieza(f"inyector_{n}", f"Inyector {n}", "Pulveriza el gasoil dentro del cilindro, a muy alta presión.",
                        "inyector", u)
                a.agregar("acero", mover(cilindro(0.07 * D, 1.0 * D, 16), (x + desfase[b], tope_culata + 0.1 * D, 0)), R)
                a.agregar("negro", mover(caja(0.22 * D, 0.25 * D, 0.22 * D), (x + desfase[b], tope_culata + 0.65 * D, 0)), R)
            else:
                a.pieza(f"bujia_{n}", f"Bujía {n}", "Hace saltar la chispa que enciende la mezcla en el cilindro.",
                        "bujia", u)
                a.agregar("ceramica", mover(cilindro(0.07 * D, 0.7 * D, 16), (x + desfase[b], tope_culata + 0.25 * D, 0)), R)
                a.agregar("acero", mover(cilindro(0.11 * D, 0.14 * D, 6, plano=True),
                                         (x + desfase[b], tope_culata + 0.0 * D, 0)), R)
        a.pieza(f"tapa_valvulas_{b}", f"Tapa de válvulas{lado(b)}",
                "Cubre los árboles de levas y las válvulas, y evita que salpique el aceite.", "tapa_de_valvulas", u)
        a.agregar("rojo" if c.combustible == "nafta" else "negro",
                  mover(caja(largo - 0.1 * D, 0.42 * D, 1.15 * D), (desfase[b], tope_culata + 0.21 * D + 0.12 * D, 0)), R)

    # Distribución (adelante): engranajes del cigüeñal y de las levas, y la correa o cadena que los une
    x_dist = largo / 2 + 0.05 * D
    levas = []                                    # (banco, z local del árbol) de cada árbol de levas
    for b in range(len(bancos)):
        zs = [0.0] if c.arboles == "sohc" else [-0.24 * D, 0.24 * D]
        levas += [(b, z) for z in zs]
    centros = [np.array([0.0, 0.0])]              # en el plano YZ: el cigüeñal y cada árbol de levas
    for b, z in levas:
        R = _hacia(bancos[b])
        p = np.array([0.0, tope_culata + 0.12 * D, z], np.float32) @ R.T
        centros.append(np.array([p[1], p[2]]))
    cadena = c.distribucion == "cadena"
    a.pieza("distribucion_correa", "Cadena de distribución" if cadena else "Correa de distribución",
            "Sincroniza el cigüeñal con los árboles de levas: las válvulas se abren en el momento justo.",
            "correa_distribucion", (1, 0, 0))
    radios = [0.32 * D] + [0.55 * D] * len(levas)
    orden_c = sorted(range(len(centros)), key=lambda i: math.atan2(*(centros[i] - np.mean(centros, 0))[::-1]))
    for i, j in zip(orden_c, orden_c[1:] + orden_c[:1]):
        p, q = centros[i], centros[j]
        d = q - p
        largo_tramo = float(np.linalg.norm(d))
        if largo_tramo < 1e-6:
            continue
        normal = np.array([d[1], -d[0]]) / largo_tramo
        if np.dot(normal, (p + q) / 2 - np.mean(centros, 0)) < 0:
            normal = -normal
        medio = (p + q) / 2 + normal * (radios[i] + radios[j]) / 2
        ang = math.atan2(d[1], d[0])
        a.agregar("negro" if not cadena else "acero",
                  mover(caja(0.35 * D, largo_tramo, 0.07 * D), (x_dist + 0.02 * D, *medio), rot_x(ang)))
    a.pieza("distribucion_poleas", "Engranajes de distribución" if cadena else "Poleas de distribución",
            "Ruedas del cigüeñal y de cada árbol de levas por las que pasa la " + ("cadena." if cadena else "correa."),
            "poleas_distribucion", (1, 0, 0))
    for (p, rad) in zip(centros, radios):
        perfil = perfil_engranaje(18 if rad < 0.4 * D else 30, rad * 0.9, rad)
        a.agregar("acero", mover(prisma_anular(perfil, 0.1 * D, 0.3 * D), (x_dist, *p), EJE_X))
    a.pieza("polea_ciguenal", "Polea del cigüeñal", "Mueve con una correa el alternador y la bomba de agua.",
            "polea_ciguenal", (1, 0, 0))
    a.mover(tipo="giro", eje=(1, 0, 0), centro=(0, 0, 0), relacion=1)
    a.agregar("hierro", mover(cilindro(0.72 * D, 0.28 * D, 36), (largo / 2 + 0.55 * D, 0, 0), EJE_X))
    a.agregar("goma", mover(toroide(0.72 * D, 0.05 * D, 36, 8), (largo / 2 + 0.55 * D, 0, 0), EJE_X))

    for i, (b, z) in enumerate(levas):
        R, u = _hacia(bancos[b]), _dir(bancos[b])
        tipo = "" if c.arboles == "sohc" else (" de admisión" if z * lado_admision[b] > 0 else " de escape")
        a.pieza(f"arbol_levas_{i}", f"Árbol de levas{tipo}{lado(b)}",
                "Eje con levas que empujan las válvulas para abrirlas, sincronizado con el cigüeñal.",
                "arbol_de_levas", u)
        a.mover(tipo="giro", eje=(1, 0, 0), relacion=0.5,   # media vuelta por cada vuelta del cigüeñal
                centro=np.array([desfase[b], tope_culata + 0.12 * D, z], np.float32) @ R.T)
        a.agregar("acero", mover(cilindro(0.07 * D, largo, 16), (desfase[b], tope_culata + 0.12 * D, z), EJE_X), R)
        for k, x in enumerate(xs):
            for dx in ((0.0,) if c.valvulas == 2 else (-0.2 * D, 0.2 * D)):
                giro_leva = rot_x(0.9 * k + (0.6 if dx > 0 else 0))
                leva = mover(mover(caja(0.12 * D, 0.26 * D, 0.17 * D), (0, 0.05 * D, 0)), R=giro_leva)
                a.agregar("acero", mover(leva, (x + dx + desfase[b], tope_culata + 0.12 * D, z)), R)

    # ---------------------------------------------------------------- 3. válvulas, culata y junta
    for b, fi in enumerate(bancos):
        R, u = _hacia(fi), _dir(fi)
        for k, x in enumerate(xs):
            n = numero(b, k)
            a.pieza(f"valvulas_{n}", f"Válvulas del cilindro {n}",
                    f"{c.valvulas} válvulas con sus resortes: dejan entrar el aire y salir los gases.", "valvulas", u)
            for dx in ((0.0,) if c.valvulas == 2 else (-0.2 * D, 0.2 * D)):
                for z in (-0.24 * D, 0.24 * D):
                    centro_x = x + dx + desfase[b]
                    cabeza = 0.15 * D if c.valvulas == 4 else 0.2 * D
                    a.agregar("acero", mover(cilindro(0.035 * D, 0.85 * D, 10), (centro_x, cubierta + 0.55 * D, z)), R)
                    a.agregar("acero", mover(cilindro(cabeza, 0.04 * D, 20), (centro_x, cubierta + 0.15 * D, z)), R)
                    a.agregar("cobre", mover(cilindro(0.085 * D, 0.28 * D, 12), (centro_x, cubierta + 0.8 * D, z)), R)
        a.pieza(f"culata_{b}", f"Culata{lado(b)}",
                "Cierra los cilindros por arriba: tiene las cámaras de combustión, los conductos y las válvulas.",
                "culata", u)
        a.agregar("aluminio", mover(caja(largo, alto_culata, 1.45 * D), (desfase[b], cubierta + 0.12 + alto_culata / 2, 0)), R)
        a.pieza(f"junta_culata_{b}", f"Junta de culata{lado(b)}",
                "Lámina que sella la unión entre la culata y el bloque.", "junta_culata", u)
        a.agregar("cobre", mover(caja(largo, 0.12, 1.45 * D), (desfase[b], cubierta + 0.06, 0)), R)

    # ---------------------------------------------------------------- 4. abajo: cárter y bomba de aceite
    a.pieza("carter", "Cárter", "Bandeja de abajo del motor: guarda el aceite.", "carter", (0, -1, 0))
    a.agregar("negro", mover(caja(largo, 0.9 * D, 2 * min(ancho_total, 1.4 * D)), (0, -r - 0.95 * D, 0)))
    a.pieza("bomba_aceite", "Bomba de aceite", "Lleva el aceite a presión a todas las piezas que se mueven.",
            "bomba_aceite", (0, -1, 0))
    a.agregar("aluminio", mover(caja(0.6 * D, 0.45 * D, 0.7 * D), (largo / 2 - 0.5 * D, -r - 0.35 * D, 0)))
    a.agregar("acero", mover(cilindro(0.08 * D, 0.6 * D, 12), (largo / 2 - 0.5 * D, -r - 0.8 * D, 0)))

    # ---------------------------------------------------------------- 5. tren alternativo: pistones y bielas
    for b, fi in enumerate(bancos):
        R, u = _hacia(fi), np.array(_dir(fi))
        w = np.array([0.0, -u[2], u[1]])
        for k, x in enumerate(xs):
            n = numero(b, k)
            munon = np.array([0.0, r * math.cos(giro[k]), r * math.sin(giro[k])])
            a_lo_largo, de_costado = float(np.dot(munon, u)), float(np.dot(munon, w))
            perno = a_lo_largo + math.sqrt(max(L ** 2 - de_costado ** 2, 1e-6))
            xb = x + desfase[b]
            a.pieza(f"piston_{n}", f"Pistón {n}", "Sube y baja dentro del cilindro; recibe la fuerza de la combustión.",
                    "piston", u)
            a.mover(tipo="piston", u=u, r=r, L=L, fase=giro[k], s0=perno, x=xb)
            a.agregar("aluminio", mover(cilindro(0.48 * D, 0.62 * D, 32), (xb, perno + 0.15 * D, 0)), R)
            for dy in (0.32, 0.4):
                a.agregar("hierro", mover(toroide(0.48 * D, 0.018 * D, 32, 6), (xb, perno + dy * D, 0)), R)
            a.agregar("acero", mover(cilindro(0.1 * D, 0.75 * D, 14), (xb, perno, 0), EJE_X), R)
            a.pieza(f"biela_{n}", f"Biela {n}", "Une el pistón con el cigüeñal y convierte la subida y bajada en giro.",
                    "biela", tuple(u))
            a.mover(tipo="biela", u=u, r=r, L=L, fase=giro[k], s0=perno, x=xb)
            # en el sistema del banco: del muñón (a_lo_largo, de_costado) al perno (perno, 0)
            dy, dz = perno - a_lo_largo, -de_costado
            largo_biela = math.hypot(dy, dz)
            a.agregar("acero", mover(mover(caja(0.16 * D, largo_biela, 0.11 * D), R=rot_x(math.atan2(dz, dy))),
                                     (xb, (perno + a_lo_largo) / 2, de_costado / 2)), R)
            a.agregar("acero", mover(toroide(0.2 * D, 0.06 * D, 24, 8), (xb, a_lo_largo, de_costado), EJE_X), R)
            a.agregar("acero", mover(toroide(0.11 * D, 0.04 * D, 20, 8), (xb, perno, 0), EJE_X), R)

    # ---------------------------------------------------------------- 6. volante y cigüeñal
    a.pieza("volante_inercia", "Volante de inercia",
            "Disco pesado que suaviza el giro; su corona dentada engrana con el motor de arranque.",
            "volante_de_inercia", (-1, 0, 0))
    a.mover(tipo="giro", eje=(1, 0, 0), centro=(0, 0, 0), relacion=1)
    x_volante = -largo / 2 - 0.35 * D
    a.agregar("hierro", mover(cilindro(1.5 * D, 0.22 * D, 48), (x_volante, 0, 0), EJE_X))
    a.agregar("acero", mover(prisma_anular(perfil_engranaje(72, 1.5 * D, 1.6 * D), 1.45 * D, 0.18 * D),
                             (x_volante, 0, 0), EJE_X))
    a.pieza("ciguenal", "Cigüeñal", "Eje que gira con el empuje de las bielas: de él sale la fuerza del motor.",
            "ciguenal", (0, -1, 0))
    a.mover(tipo="giro", eje=(1, 0, 0), centro=(0, 0, 0), relacion=1)
    a.agregar("acero", mover(cilindro(0.26 * D, largo + 0.5 * D, 28), (0, 0, 0), EJE_X))
    ancho_munon = 0.3 * D if len(bancos) == 1 else 0.62 * D
    for k, x in enumerate(xs):
        y, z = r * math.cos(giro[k]), r * math.sin(giro[k])
        a.agregar("acero", mover(cilindro(0.2 * D, ancho_munon, 20), (x, y, z), EJE_X))
        for lado_x in (-1, 1):
            xm = x + lado_x * (ancho_munon / 2 + 0.06 * D)
            brazo = mover(caja(0.12 * D, r + 0.5 * D, 0.55 * D), (0, (r + 0.5 * D) / 2 - 0.25 * D, 0))
            a.agregar("acero", mover(brazo, (xm, 0, 0), rot_x(giro[k])))
            contrapeso = mover(caja(0.12 * D, 0.55 * r, 0.9 * D), (0, -0.45 * r, 0))
            a.agregar("hierro", mover(contrapeso, (xm, 0, 0), rot_x(giro[k])))

    # ---------------------------------------------------------------- 7. el bloque queda fijo
    a.pieza("bloque", "Bloque", "Cuerpo principal del motor: contiene los cilindros y sostiene el cigüeñal.",
            "bloque", fija=True)
    for b, fi in enumerate(bancos):
        R = _hacia(fi)
        alto = cubierta - 0.35 * r
        a.agregar("hierro" if c.combustible == "diesel" else "aluminio",
                  mover(caja(largo, alto, 1.4 * D), (desfase[b], 0.35 * r + alto / 2, 0)), R)
    a.agregar("hierro" if c.combustible == "diesel" else "aluminio",
              mover(caja(largo, 2 * r + 0.6 * D, 2 * min(ancho_total, 1.3 * D)), (0, -0.2 * r, 0)))
    return m


def _radial(c: Combustion) -> Modelo:
    """Motor radial de aviación: una estrella de cilindros alrededor del cigüeñal, con la hélice adelante."""
    m = Modelo(MATERIALES)
    m.velocidad = VELOCIDAD
    a = _Armado(m)
    D, r = c.diametro, c.carrera / 2
    L = 1.9 * c.carrera
    base = r + 0.9 * D                           # donde arranca cada cilindro, desde el eje
    tope = r + L + 0.6 * D
    angulos = [2 * math.pi * k / c.cilindros for k in range(c.cilindros)]

    a.pieza("helice", "Hélice", "Convierte el giro del motor en empuje.", "helice", (1, 0, 0))
    a.mover(tipo="giro", eje=(1, 0, 0), centro=(0, 0, 0), relacion=0.5)   # la caja de reducción la hace ir más lenta
    x_helice = 1.9 * D
    a.agregar("hierro", mover(cono(0.45 * D, 0.05 * D, 0.9 * D, 24), (x_helice + 0.45 * D, 0, 0), rot_z(-math.pi / 2)))
    for ang in (0, math.pi):
        pala = mover(caja(0.12 * D, 3.2 * tope / 2, 0.5 * D), (0, 3.2 * tope / 4 + 0.3 * D, 0), rot_x(ang))
        a.agregar("madera", mover(pala, (x_helice, 0, 0)))
    a.pieza("tapa_delantera", "Caja de reducción", "Tapa delantera con los engranajes que bajan las vueltas de la hélice.",
            "tapa_delantera", (1, 0, 0))
    a.agregar("aluminio", mover(cono(1.1 * base, 0.45 * D, 1.2 * D, 36), (1.0 * D, 0, 0), rot_z(-math.pi / 2)))
    a.pieza("multiple_escape", "Colector de escape", "Anillo que junta los gases de todos los cilindros.",
            "multiple_escape", (-1, 0, 0))
    a.agregar("escape", mover(toroide(0.75 * tope, 0.14 * D, 64, 10), (-0.9 * D, 0, 0), EJE_X))
    for ang in angulos:
        R = _hacia(ang)
        a.agregar("escape", mover(cilindro(0.1 * D, 1.0 * D, 12), (-0.5 * D, 0.75 * tope, 0), rot_z(math.pi / 3)), R)
    for k, ang in enumerate(angulos):
        R, u = _hacia(ang), _dir(ang)
        a.pieza(f"bujias_{k + 1}", f"Bujías del cilindro {k + 1}", "Dos bujías por cilindro, por seguridad en vuelo.",
                "bujia", u)
        for z in (-0.3 * D, 0.3 * D):
            a.agregar("ceramica", mover(cilindro(0.07 * D, 0.5 * D, 12), (0, tope + 0.2 * D, z)), R)
    for k, ang in enumerate(angulos):
        R, u = _hacia(ang), _dir(ang)
        a.pieza(f"culata_{k + 1}", f"Culata {k + 1}", "Cabeza del cilindro, con las válvulas y sus balancines.",
                "culata", u)
        a.agregar("aluminio", mover(cilindro(0.62 * D, 0.55 * D, 28), (0, tope - 0.25 * D, 0)), R)
        a.agregar("aluminio", mover(caja(0.5 * D, 0.3 * D, 1.0 * D), (0, tope + 0.12 * D, 0)), R)
    for k, ang in enumerate(angulos):
        R, u = _hacia(ang), _dir(ang)
        a.pieza(f"cilindro_{k + 1}", f"Cilindro {k + 1}", "Camisa con aletas de refrigeración por aire.",
                "cilindro", u)
        alto = tope - 0.5 * D - base
        a.agregar("hierro", mover(cilindro(0.58 * D, alto, 28), (0, base + alto / 2, 0)), R)
        for f in range(7):
            a.agregar("hierro", mover(cilindro(0.78 * D, 0.04 * D, 28), (0, base + 0.15 * D + f * alto / 7.5, 0)), R)
    for k, ang in enumerate(angulos):
        R, u = _hacia(ang), _dir(ang)
        perno = r + L
        a.pieza(f"piston_{k + 1}", f"Pistón {k + 1}", "Sube y baja dentro del cilindro.", "piston", u)
        # el muñón único apunta arriba con el cigüeñal en 0; cada cilindro lo ve girado su propio ángulo
        a.mover(tipo="piston", u=u, r=r, L=L, fase=0.0, s0=perno, x=0.0)
        a.agregar("aluminio", mover(cilindro(0.48 * D, 0.62 * D, 28), (0, perno + 0.15 * D, 0)), R)
        a.agregar("acero", mover(cilindro(0.1 * D, 0.75 * D, 12), (0, perno, 0), EJE_Z), R)
        maestra = k == 0
        a.pieza(f"biela_{k + 1}", "Biela maestra" if maestra else f"Biela articulada {k + 1}",
                "La biela maestra va al cigüeñal; las articuladas se enganchan a ella." if maestra else
                "Se engancha a la biela maestra.", "biela", u)
        # la maestra sigue al muñón; las articuladas, simplificado, acompañan a su pistón
        a.mover(tipo="biela" if maestra else "piston", u=u, r=r, L=L, fase=0.0, s0=perno, x=0.0)
        desde = r if maestra else r + 0.45 * D
        a.agregar("acero", mover(caja(0.11 * D, perno - desde, 0.16 * D), (0, (perno + desde) / 2, 0)), R)
        if maestra:   # el anillo de la biela maestra abraza el muñón del cigüeñal
            a.agregar("acero", mover(toroide(0.55 * D, 0.1 * D, 36, 8), (0, r, 0), EJE_X))
    a.pieza("ciguenal", "Cigüeñal", "De un solo codo: todas las bielas empujan el mismo muñón.", "ciguenal", (1, 0, 0))
    a.mover(tipo="giro", eje=(1, 0, 0), centro=(0, 0, 0), relacion=1)
    a.agregar("acero", mover(cilindro(0.3 * D, 3.0 * D, 24), (0.6 * D, 0, 0), EJE_X))
    a.agregar("acero", mover(cilindro(0.22 * D, 0.5 * D, 20), (0, r, 0), EJE_X))
    a.agregar("hierro", mover(caja(0.14 * D, r + 0.6 * D, 1.0 * D), (0.32 * D, 0.2 * r, 0)))
    a.pieza("bloque", "Cárter central", "Cuerpo central donde se atornillan los cilindros y gira el cigüeñal.",
            "bloque", fija=True)
    a.agregar("aluminio", mover(cilindro(base + 0.05 * D, 1.5 * D, 48), (0, 0, 0), EJE_X))
    a.agregar("aluminio", mover(cono(base * 0.95, base * 0.5, 1.2 * D, 36), (-1.3 * D, 0, 0), rot_z(math.pi / 2)))
    return m
