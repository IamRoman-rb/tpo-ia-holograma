"""Geometría 3D y escritura de .glb por piezas (sin dependencias más allá de numpy).

La usan el catálogo de demostración (scripts/generar_modelos_demo.py) y la generación
de objetos por piezas con el LLM (backend/ensamble.py).

Cada pieza se escribe como un nodo separado del .glb, con sus metadatos en "extras"
(nombre, descripción, orden y dirección de desmontaje). Three.js los lee en object.userData,
y así el visor puede desmontar el objeto pieza por pieza.
"""
import json
import math
import struct
from collections import defaultdict

import numpy as np

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


def cono(r_base, r_tope, h, seg=48):
    """Tronco de cono sobre el eje Y (r_tope = 0 da un cono)."""
    p, n, i = [], [], []
    y0, y1 = -h / 2, h / 2
    pendiente = (r_base - r_tope) / h if h else 0
    largo = math.hypot(1, pendiente)
    for k in range(seg + 1):
        a = 2 * math.pi * k / seg
        c, s = math.cos(a), math.sin(a)
        normal = (c / largo, pendiente / largo, s / largo)
        p += [(r_base * c, y0, r_base * s), (r_tope * c, y1, r_tope * s)]
        n += [normal, normal]
    for k in range(seg):
        b = 2 * k
        i += [(b, b + 1, b + 2), (b + 1, b + 3, b + 2)]
    for y, ny, r in ((y0, -1, r_base), (y1, 1, r_tope)):
        if r <= 0:
            continue
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


def rotacion_grados(rx, ry, rz):
    """Matriz de rotación a partir de ángulos en grados (primero X, después Y, después Z)."""
    return rot_z(math.radians(rz)) @ rot_y(math.radians(ry)) @ rot_x(math.radians(rx))


# ---------------------------------------------------------------- modelo por piezas

class Modelo:
    """Un objeto formado por piezas. Cada pieza agrupa partes geométricas por material."""

    def __init__(self, materiales):
        self.materiales = materiales          # nombre -> (rgb, metálico, rugosidad)
        self.piezas = {}                      # id -> {"meta": {...}, "partes": {material: [partes]}}
        self._actual = None

    def pieza(self, id_, nombre, descripcion="", orden=None, direccion=None):
        """Empieza una pieza nueva: lo que se agregue después pertenece a ella."""
        self.piezas[id_] = {
            "meta": {"pieza": id_, "nombre": nombre, "descripcion": descripcion,
                     "orden": orden, "direccion": list(direccion) if direccion is not None else None},
            "partes": defaultdict(list),
        }
        self._actual = id_
        return self

    def agregar(self, material, parte):
        if self._actual is None:
            self.pieza("principal", "Pieza principal")
        self.piezas[self._actual]["partes"][material].append(parte)


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


def _unir(partes):
    P, Nn, I, base = [], [], [], 0
    for p, n, i in partes:
        P.append(p)
        Nn.append(n)
        I.append(i + base)
        base += len(p)
    P, Nn, I = np.concatenate(P), np.concatenate(Nn), np.concatenate(I)
    return P, _normalizar(Nn), _corregir_giro(P, Nn, I)


def a_glb(modelo, nombre, unidad=None) -> bytes:
    """Escribe el modelo como glTF binario: un nodo por pieza, con sus metadatos en extras.
    unidad (por ejemplo "cm") queda en los extras de la escena: el visor la usa para mostrar medidas."""
    binario = bytearray()
    vistas, accesores, materiales, mallas, nodos = [], [], [], [], []
    indice_material = {}

    def vista(datos, destino):
        vistas.append({"buffer": 0, "byteOffset": len(binario), "byteLength": len(datos), "target": destino})
        binario.extend(datos)
        while len(binario) % 4:
            binario.append(0)
        return len(vistas) - 1

    for id_, pieza in modelo.piezas.items():
        primitivas = []
        for mat, partes in pieza["partes"].items():
            if not partes:
                continue
            if mat not in indice_material:
                rgb, metal, rugosidad = modelo.materiales[mat]
                materiales.append({"name": mat, "pbrMetallicRoughness": {
                    "baseColorFactor": [*rgb, 1.0], "metallicFactor": metal, "roughnessFactor": rugosidad}})
                indice_material[mat] = len(materiales) - 1
            P, Nn, I = _unir(partes)
            accesores.append({"bufferView": vista(P.tobytes(), 34962), "componentType": 5126, "count": len(P),
                              "type": "VEC3", "min": P.min(0).tolist(), "max": P.max(0).tolist()})
            accesores.append({"bufferView": vista(Nn.tobytes(), 34962), "componentType": 5126,
                              "count": len(Nn), "type": "VEC3"})
            accesores.append({"bufferView": vista(I.astype(np.uint32).tobytes(), 34963), "componentType": 5125,
                              "count": I.size, "type": "SCALAR"})
            k = len(accesores)
            primitivas.append({"attributes": {"POSITION": k - 3, "NORMAL": k - 2}, "indices": k - 1,
                               "material": indice_material[mat]})
        if not primitivas:
            continue
        mallas.append({"name": id_, "primitives": primitivas})
        extras = {k: v for k, v in pieza["meta"].items() if v not in (None, "")}
        nodos.append({"name": id_, "mesh": len(mallas) - 1, "extras": extras})

    gltf = {
        "asset": {"version": "2.0", "generator": "tpo-holograma/backend/geometria.py"},
        "scene": 0, "scenes": [{"name": nombre, "nodes": list(range(len(nodos))), **({"extras": {"unidad": unidad}} if unidad else {})}],
        "nodes": nodos, "meshes": mallas, "materials": materiales, "accessors": accesores,
        "bufferViews": vistas, "buffers": [{"byteLength": len(binario)}],
    }
    js = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binario)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binario), b"BIN" + bytes([0])) + bytes(binario))


def escribir_glb(modelo, ruta, nombre):
    with open(ruta, "wb") as f:
        f.write(a_glb(modelo, nombre))
