"""Rasgos legibles de una mano para los árboles de decisión. Son los MISMOS que frontend/js/arbol.js
(rasgosArbol), con el mismo orden: las métricas de las reglas (gestures.js: medir) más el ángulo de cada
articulación y distancias entre puntas. frontend/tests/arbol.test.mjs verifica que den lo mismo."""
import numpy as np

RASGOS = ["pulgarIndice", "pulgarAfuera", "pulgarVertical", "separacionV", "indice", "medio", "anular",
          "menique", "curvaPulgar", "curvaIndice", "curvaMedio", "curvaAnular", "curvaMenique", "pulgarMedio",
          "pulgarAnular", "pulgarMenique", "separacionMedioAnular", "separacionAnularMenique", "indiceVertical"]


def _dist(p, a, b):
    return np.linalg.norm(p[:, a] - p[:, b], axis=1)


def _curva(p, a, b, c):
    u, v = p[:, b] - p[:, a], p[:, c] - p[:, b]
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    coseno = (u * v).sum(axis=1) / np.where(n == 0, 1, n)
    return np.degrees(np.arccos(np.clip(coseno, -1, 1)))


def _razon(num, den):
    return num / np.where(den == 0, 1, den)


def rasgos(x):
    """x: (n, 63) normalizado (preparar.normalizar). Devuelve (n, 19) en el orden de RASGOS."""
    p = np.asarray(x, dtype=np.float64).reshape(-1, 21, 3)
    columnas = {
        "pulgarIndice": _dist(p, 4, 8),
        "pulgarAfuera": _dist(p, 4, 5),
        "pulgarVertical": _razon(p[:, 4, 1] - p[:, 2, 1], _dist(p, 4, 2)),
        "separacionV": _dist(p, 8, 12),
        "indice": _razon(_dist(p, 8, 0), _dist(p, 6, 0)),
        "medio": _razon(_dist(p, 12, 0), _dist(p, 10, 0)),
        "anular": _razon(_dist(p, 16, 0), _dist(p, 14, 0)),
        "menique": _razon(_dist(p, 20, 0), _dist(p, 18, 0)),
        "curvaPulgar": _curva(p, 2, 3, 4),
        "curvaIndice": _curva(p, 5, 6, 8),
        "curvaMedio": _curva(p, 9, 10, 12),
        "curvaAnular": _curva(p, 13, 14, 16),
        "curvaMenique": _curva(p, 17, 18, 20),
        "pulgarMedio": _dist(p, 4, 12),
        "pulgarAnular": _dist(p, 4, 16),
        "pulgarMenique": _dist(p, 4, 20),
        "separacionMedioAnular": _dist(p, 12, 16),
        "separacionAnularMenique": _dist(p, 16, 20),
        "indiceVertical": _razon(p[:, 8, 1] - p[:, 5, 1], _dist(p, 5, 8)),
    }
    return np.stack([columnas[r] for r in RASGOS], axis=1).astype(np.float32)
