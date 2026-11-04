"""Lo que ve el reconocimiento de piezas de cada pieza de un modelo 3D: su forma y su lugar en el conjunto.

Se usa igual con los motores generados (para entrenar) y con los modelos reales de Objaverse (para etiquetarlos),
así la red ve lo mismo en los dos casos. Para no depender de cómo está orientado el archivo, todo se expresa en los
ejes principales del conjunto (PCA de los puntos de todas las piezas).

Por pieza:
  - puntos: N puntos de su superficie (al azar, proporcional al área), centrados en la pieza y escalados por su
    propio tamaño: la forma, sin importar dónde está ni qué tan grande es;
  - contexto (CONTEXTO valores): dónde está y qué tan grande es respecto del conjunto, cuánto se parece a una barra,
    a un disco o a una bola (autovalores de su PCA), qué fracción del área total ocupa y cuántas piezas parecidas
    tiene (los pistones vienen de a 4, 6, 8…).
"""
import numpy as np

PUNTOS = 512
CONTEXTO = 15


def muestrear(posiciones, triangulos, n, rng):
    """n puntos al azar sobre la superficie (proporcional al área de cada triángulo). Devuelve (puntos, área)."""
    a, b, c = (posiciones[triangulos[:, k]] for k in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if total <= 0 or not np.isfinite(total):
        return np.repeat(posiciones[:1], n, axis=0), 0.0
    elegidos = rng.choice(len(areas), n, p=areas / total)
    u, v = rng.random((2, n, 1))
    fuera = (u + v) > 1
    u, v = np.where(fuera, 1 - u, u), np.where(fuera, 1 - v, v)
    return a[elegidos] + u * (b[elegidos] - a[elegidos]) + v * (c[elegidos] - a[elegidos]), total


def _ejes(puntos):
    """Centro y ejes principales (de mayor a menor varianza) de una nube de puntos."""
    centro = puntos.mean(0)
    valores, vectores = np.linalg.eigh(np.cov((puntos - centro).T) + 1e-12 * np.eye(3))
    orden = np.argsort(valores)[::-1]
    return centro, np.maximum(valores[orden], 0), vectores[:, orden]


def rasgos(piezas, rng, n=PUNTOS):
    """piezas: lista de (posiciones, triángulos) en coordenadas del modelo. Devuelve (puntos [P, n, 3], contexto
    [P, CONTEXTO]) en float32."""
    muestras, areas = zip(*(muestrear(p, t, n, rng) for p, t in piezas))
    todos = np.concatenate(muestras)
    centro, valores, ejes = _ejes(todos)
    escala = float(np.sqrt(valores[0])) * 2 or 1.0
    # Los signos de los ejes principales son arbitrarios: se fijan para que la mayor parte del área quede del
    # lado positivo (así un mismo motor girado da los mismos números)
    proyectado = (todos - centro) @ ejes
    ejes = ejes * np.where(np.mean(proyectado ** 3, 0) < 0, -1, 1)
    area_total = sum(areas) or 1.0

    puntos, contexto, firmas = [], [], []
    for m, area in zip(muestras, areas):
        local = (m - centro) @ ejes / escala               # en los ejes del conjunto
        c, v, _ = _ejes(local)
        tam = float(np.sqrt(v[0])) * 2 or 1e-6
        puntos.append((local - c) / tam)
        lineal = (v[0] - v[1]) / (v[0] + 1e-12)
        plano = (v[1] - v[2]) / (v[0] + 1e-12)
        bola = v[2] / (v[0] + 1e-12)
        extension = np.sort(np.ptp(local, 0))[::-1]
        contexto.append([*c, float(np.linalg.norm(c)), float(np.linalg.norm(c[1:])), *extension,
                         lineal, plano, bola, area / area_total, tam, 0.0, 0.0])
        firmas.append(np.array([*extension, lineal, plano]))
    firmas = np.array(firmas)
    for i, f in enumerate(firmas):   # piezas "gemelas": misma forma y tamaño (pistones, bielas, bujías…)
        parecidas = np.sum(np.max(np.abs(firmas - f) / (np.abs(f) + 0.05), axis=1) < 0.15) - 1
        contexto[i][-2] = np.log1p(parecidas)
        contexto[i][-1] = np.log1p(len(piezas))
    return np.asarray(puntos, np.float32), np.asarray(contexto, np.float32)
