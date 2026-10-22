"""Genera el catálogo precargado de modelos .glb (plan B de la demo).

Construye piezas industriales simples a partir de primitivas (cilindros, toroides,
esferas, prismas) y las escribe en formato glTF binario sin dependencias externas
más allá de numpy. También genera frontend/models/catalogo.json.

Uso (desde la raíz):  python scripts/generar_modelos_demo.py
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

SALIDA = Path(__file__).resolve().parent.parent / "frontend" / "models"

# ---------------------------------------------------------------- primitivas
# Cada primitiva devuelve (posiciones, normales, índices). El sentido de giro de
# los triángulos se corrige al final según las normales, así que no importa acá.


def _parte(p, n, i):
    return (np.asarray(p, np.float32).reshape(-1, 3),
            np.asarray(n, np.float32).reshape(-1, 3),
            np.asarray(i, np.uint32).reshape(-1, 3))


def cilindro(r, h, seg=48, plano=False, tapas=True):
    """Cilindro sobre el eje Y, centrado en el origen. plano=True: caras facetadas (hexágono)."""
    p, n, i = [], [], []
    y0, y1 = -h / 2, h / 2
    if plano:
        for k in range(seg):
            a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
            am = (a0 + a1) / 2
            b = len(p)
            for a, y in ((a0, y0), (a0, y1), (a1, y0), (a1, y1)):
                p.append((r * math.cos(a), y, r * math.sin(a)))
                n.append((math.cos(am), 0, math.sin(am)))
            i += [(b, b + 1, b + 2), (b + 1, b + 3, b + 2)]
    else:
        for k in range(seg + 1):
            a = 2 * math.pi * k / seg
            c, s = math.cos(a), math.sin(a)
            p += [(r * c, y0, r * s), (r * c, y1, r * s)]
            n += [(c, 0, s), (c, 0, s)]
        for k in range(seg):
            b = 2 * k
            i += [(b, b + 1, b + 2), (b + 1, b + 3, b + 2)]
    if tapas:
        for y, ny in ((y0, -1), (y1, 1)):
            centro = len(p)
            p.append((0, y, 0))
            n.append((0, ny, 0))
            for k in range(seg + 1):
                a = 2 * math.pi * k / seg
                p.append((r * math.cos(a), y, r * math.sin(a)))
                n.append((0, ny, 0))
            for k in range(seg):
                i.append((centro, centro + 1 + k, centro + 2 + k))
    return _parte(p, n, i)


def toroide(R, r, seg=64, seg_tubo=16):
    """Toroide alrededor del eje Y."""
    p, n, i = [], [], []
    for a in range(seg + 1):
        u = 2 * math.pi * a / seg
        for b in range(seg_tubo + 1):
            v = 2 * math.pi * b / seg_tubo
            p.append(((R + r * math.cos(v)) * math.cos(u), r * math.sin(v), (R + r * math.cos(v)) * math.sin(u)))
            n.append((math.cos(v) * math.cos(u), math.sin(v), math.cos(v) * math.sin(u)))
    w = seg_tubo + 1
    for a in range(seg):
        for b in range(seg_tubo):
            k = a * w + b
            i += [(k, k + w, k + 1), (k + 1, k + w, k + w + 1)]
    return _parte(p, n, i)


def esfera(r, seg=32, anillos=16):
    p, n, i = [], [], []
    for a in range(anillos + 1):
        t = math.pi * a / anillos
        for b in range(seg + 1):
            f = 2 * math.pi * b / seg
            d = (math.sin(t) * math.cos(f), math.cos(t), math.sin(t) * math.sin(f))
            p.append(tuple(r * x for x in d))
            n.append(d)
    w = seg + 1
    for a in range(anillos):
        for b in range(seg):
            k = a * w + b
            i += [(k, k + w, k + 1), (k + 1, k + w, k + w + 1)]
    return _parte(p, n, i)


def caja(dx, dy, dz):
    p, n, i = [], [], []
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    for eje in range(3):
        for signo in (-1, 1):
            normal = [0, 0, 0]
            normal[eje] = signo
            u, v = [(1, 2), (0, 2), (0, 1)][eje]
            b = len(p)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                q = [0, 0, 0]
                q[eje] = signo
                q[u], q[v] = su, sv
                p.append((q[0] * hx, q[1] * hy, q[2] * hz))
                n.append(normal)
            i += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return _parte(p, n, i)


def prisma_anular(perfil, r_agujero, h):
    """Extruye sobre Y un contorno exterior (lista de (x, z)) con un agujero circular centrado."""
    p, n, i = [], [], []
    N = len(perfil)
    y0, y1 = -h / 2, h / 2
    interior = [(r_agujero * math.cos(math.atan2(z, x)), r_agujero * math.sin(math.atan2(z, x))) for x, z in perfil]
    # caras superior e inferior: franja entre el contorno y el agujero
    for y, ny in ((y0, -1), (y1, 1)):
        b = len(p)
        for (xo, zo), (xi, zi) in zip(perfil, interior):
            p += [(xo, y, zo), (xi, y, zi)]
            n += [(0, ny, 0), (0, ny, 0)]
        for k in range(N):
            a, c = b + 2 * k, b + 2 * ((k + 1) % N)
            i += [(a, a + 1, c), (a + 1, c + 1, c)]
    # pared exterior facetada
    for k in range(N):
        (xa, za), (xb, zb) = perfil[k], perfil[(k + 1) % N]
        nx, nz = zb - za, -(xb - xa)
        largo = math.hypot(nx, nz) or 1
        nx, nz = nx / largo, nz / largo
        if nx * (xa + xb) + nz * (za + zb) < 0:  # que apunte hacia afuera
            nx, nz = -nx, -nz
        b = len(p)
        p += [(xa, y0, za), (xa, y1, za), (xb, y0, zb), (xb, y1, zb)]
        n += [(nx, 0, nz)] * 4
        i += [(b, b + 1, b + 2), (b + 1, b + 3, b + 2)]
    # pared del agujero, normales hacia el centro
    b = len(p)
    for xi, zi in interior + interior[:1]:
        d = (-xi / r_agujero, 0, -zi / r_agujero)
        p += [(xi, y0, zi), (xi, y1, zi)]
        n += [d, d]
    for k in range(N):
        a = b + 2 * k
        i += [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
    return _parte(p, n, i)


def circulo(r, seg=64):
    return [(r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg)) for k in range(seg)]


def perfil_engranaje(dientes, r_raiz, r_punta):
    pts, paso = [], 2 * math.pi / dientes
    for d in range(dientes):
        a = d * paso
        for frac, r in ((0, r_raiz), (0.15, r_punta), (0.45, r_punta), (0.6, r_raiz)):
            pts.append((r * math.cos(a + frac * paso), r * math.sin(a + frac * paso)))
    return pts


# ---------------------------------------------------------------- transformaciones

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def mover(parte, t=(0, 0, 0), R=None):
    p, n, i = parte
    if R is not None:
        p, n = p @ R.T, n @ R.T
    return p + np.asarray(t, np.float32), n, i


# ---------------------------------------------------------------- modelo y escritura glb

class Modelo:
    def __init__(self, materiales):
        self.materiales = materiales          # nombre -> (rgb, metálico, rugosidad)
        self.partes = {m: [] for m in materiales}

    def agregar(self, material, parte):
        self.partes[material].append(parte)

    def primitivas(self):
        for mat, partes in self.partes.items():
            if not partes:
                continue
            P, Nn, I, base = [], [], [], 0
            for p, n, i in partes:
                P.append(p)
                Nn.append(n)
                I.append(i + base)
                base += len(p)
            P, Nn, I = np.concatenate(P), np.concatenate(Nn), np.concatenate(I)
            yield mat, P, _normalizar(Nn), _corregir_giro(P, Nn, I)


def _normalizar(n):
    largo = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.where(largo == 0, 1, largo)).astype(np.float32)


def _corregir_giro(P, N, I):
    """glTF usa caras frontales antihorarias: invierte los triángulos cuyo giro contradice sus normales."""
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    cara = np.cross(b - a, c - a)
    media = N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]]
    invertir = np.einsum("ij,ij->i", cara, media) < 0
    I = I.copy()
    I[invertir] = I[invertir][:, [0, 2, 1]]
    return I


def escribir_glb(modelo, ruta, nombre):
    binario = bytearray()
    vistas, accesores, prims, mats = [], [], [], []

    def vista(datos, destino):
        vistas.append({"buffer": 0, "byteOffset": len(binario), "byteLength": len(datos), "target": destino})
        binario.extend(datos)
        while len(binario) % 4:
            binario.append(0)
        return len(vistas) - 1

    for mat, P, Nn, I in modelo.primitivas():
        rgb, metal, rugosidad = modelo.materiales[mat]
        mats.append({"name": mat, "pbrMetallicRoughness": {
            "baseColorFactor": [*rgb, 1.0], "metallicFactor": metal, "roughnessFactor": rugosidad}})
        accesores.append({"bufferView": vista(P.tobytes(), 34962), "componentType": 5126, "count": len(P),
                          "type": "VEC3", "min": P.min(0).tolist(), "max": P.max(0).tolist()})
        accesores.append({"bufferView": vista(Nn.tobytes(), 34962), "componentType": 5126, "count": len(Nn),
                          "type": "VEC3"})
        accesores.append({"bufferView": vista(I.astype(np.uint32).tobytes(), 34963), "componentType": 5125,
                          "count": I.size, "type": "SCALAR"})
        k = len(accesores)
        prims.append({"attributes": {"POSITION": k - 3, "NORMAL": k - 2}, "indices": k - 1,
                      "material": len(mats) - 1})

    gltf = {
        "asset": {"version": "2.0", "generator": "tpo-holograma/generar_modelos_demo.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": nombre}],
        "meshes": [{"name": nombre, "primitives": prims}],
        "materials": mats, "accessors": accesores, "bufferViews": vistas,
        "buffers": [{"byteLength": len(binario)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binario)
    with open(ruta, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(binario), b"BIN\x00") + bytes(binario))


# ---------------------------------------------------------------- piezas del catálogo

ACERO = ((0.80, 0.80, 0.82), 1.0, 0.25)
HIERRO = ((0.25, 0.27, 0.30), 0.7, 0.45)
CROMO = ((0.92, 0.92, 0.94), 1.0, 0.08)
HORIZ_X = rot_z(math.pi / 2)  # lleva un cilindro del eje Y al eje X


def valvula():
    m = Modelo({"hierro": HIERRO, "volante": ((0.75, 0.08, 0.06), 0.2, 0.4), "acero": ACERO,
                "bulones": ((0.55, 0.55, 0.55), 0.9, 0.35)})
    m.agregar("hierro", mover(cilindro(0.32, 1.2), R=HORIZ_X))
    m.agregar("hierro", esfera(0.42))
    for x in (-0.6, 0.6):
        m.agregar("hierro", mover(cilindro(0.55, 0.1), (x, 0, 0), HORIZ_X))
        for k in range(8):
            a = 2 * math.pi * k / 8
            m.agregar("bulones", mover(cilindro(0.04, 0.16, 12), (x, 0.45 * math.cos(a), 0.45 * math.sin(a)), HORIZ_X))
    m.agregar("hierro", mover(cilindro(0.32, 0.08), (0, 0.32, 0)))
    m.agregar("hierro", mover(cilindro(0.22, 0.5), (0, 0.55, 0)))
    m.agregar("acero", mover(cilindro(0.05, 0.6, 16), (0, 1.0, 0)))
    m.agregar("volante", mover(toroide(0.38, 0.035), (0, 1.28, 0)))
    m.agregar("volante", mover(cilindro(0.08, 0.08, 24), (0, 1.28, 0)))
    m.agregar("volante", mover(caja(0.76, 0.04, 0.04), (0, 1.28, 0)))
    m.agregar("volante", mover(caja(0.04, 0.04, 0.76), (0, 1.28, 0)))
    return m


def engranaje():
    m = Modelo({"acero": ACERO})
    frente = rot_x(math.pi / 2)  # queda de frente a la cámara
    m.agregar("acero", mover(prisma_anular(perfil_engranaje(24, 0.8, 0.95), 0.2, 0.25), R=frente))
    m.agregar("acero", mover(prisma_anular(circulo(0.35), 0.2, 0.45), R=frente))
    return m


def tornillo():
    m = Modelo({"zinc": ((0.72, 0.74, 0.76), 0.9, 0.3), "laton": ((0.78, 0.60, 0.25), 0.9, 0.3)})
    m.agregar("zinc", mover(cilindro(0.32, 0.22, 6, plano=True), (0, 0.9, 0)))
    m.agregar("zinc", mover(prisma_anular(circulo(0.4), 0.17, 0.04), (0, 0.77, 0)))
    m.agregar("zinc", cilindro(0.15, 1.5, 32))
    for k in range(21):
        m.agregar("zinc", mover(toroide(0.15, 0.025, 32, 8), (0, -0.7 + 0.06 * k, 0)))
    m.agregar("laton", mover(cilindro(0.3, 0.22, 6, plano=True), (0, -0.35, 0)))
    return m


def rodamiento():
    m = Modelo({"acero": ACERO, "bolas": CROMO})
    frente = rot_x(math.pi / 2)
    m.agregar("acero", mover(prisma_anular(circulo(1.0, 96), 0.82, 0.35), R=frente))
    m.agregar("acero", mover(prisma_anular(circulo(0.5, 96), 0.3, 0.35), R=frente))
    for k in range(12):
        a = 2 * math.pi * k / 12
        m.agregar("bolas", mover(esfera(0.15), (0.66 * math.cos(a), 0.66 * math.sin(a), 0)))
    return m


def motor():
    m = Modelo({"carcasa": ((0.10, 0.30, 0.60), 0.3, 0.5), "acero": ACERO, "base": HIERRO})
    m.agregar("carcasa", mover(cilindro(0.55, 1.3), R=HORIZ_X))
    for k in range(16):
        a = 2 * math.pi * k / 16
        m.agregar("carcasa", mover(caja(1.1, 0.12, 0.04), (0, 0.6 * math.cos(a), 0.6 * math.sin(a)), rot_x(a)))
    for x in (-0.72, 0.72):
        m.agregar("carcasa", mover(cilindro(0.5, 0.15), (x, 0, 0), HORIZ_X))
    m.agregar("acero", mover(cilindro(0.07, 0.5, 24), (1.0, 0, 0), HORIZ_X))
    m.agregar("carcasa", mover(caja(0.35, 0.25, 0.35), (0, 0.62, 0)))
    m.agregar("base", mover(caja(1.0, 0.1, 0.9), (0, -0.75, 0)))
    for x in (-0.35, 0.35):
        m.agregar("base", mover(caja(0.15, 0.25, 0.7), (x, -0.6, 0)))
    return m


CATALOGO = [
    ("valvula_compuerta", valvula, "Válvula de compuerta",
     ["válvula", "valvula de acero", "llave de paso", "válvula industrial", "válvula con volante"],
     "Válvula de compuerta bridada con volante de accionamiento."),
    ("engranaje_recto", engranaje, "Engranaje recto",
     ["engranaje", "rueda dentada", "piñón", "engrane", "corona dentada"],
     "Engranaje recto de 24 dientes con cubo central."),
    ("tornillo_hexagonal", tornillo, "Tornillo hexagonal con tuerca",
     ["tornillo", "bulón", "perno", "tuerca", "tornillo con tuerca"],
     "Tornillo de cabeza hexagonal con arandela y tuerca."),
    ("rodamiento_bolas", rodamiento, "Rodamiento de bolas",
     ["rodamiento", "ruleman", "rulemán", "cojinete", "balero"],
     "Rodamiento rígido de una hilera de bolas."),
    ("motor_electrico", motor, "Motor eléctrico",
     ["motor", "motor trifásico", "motor industrial", "motor con aletas", "motor eléctrico industrial"],
     "Motor eléctrico con carcasa aletada, eje y base de montaje."),
]


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    indice = []
    for id_, fabricar, nombre, sinonimos, descripcion in CATALOGO:
        ruta = SALIDA / f"{id_}.glb"
        escribir_glb(fabricar(), ruta, id_)
        indice.append({"id": id_, "nombre": nombre, "sinonimos": sinonimos, "descripcion": descripcion,
                       "url": f"models/{id_}.glb", "origen": "catalogo"})
        print(f"{ruta.name:28s} {ruta.stat().st_size / 1024:7.1f} KB")
    with open(SALIDA / "catalogo.json", "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)
    print(f"catalogo.json con {len(indice)} modelos")


if __name__ == "__main__":
    main()
