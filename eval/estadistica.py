"""Métodos estadísticos para comparar modelos con honestidad (los usan eval/ y ml/).

- Intervalo de confianza del 95 % por bootstrap: con 38 frases, 87 % puede ser cualquier cosa entre ~76 % y ~97 %.
- Test de McNemar exacto: dos modelos evaluados sobre las MISMAS frases no son muestras independientes; lo que
  importa son las frases en las que uno acierta y el otro no. La prueba binomial sobre esos casos dice si la
  diferencia puede ser casualidad.
- Calibración: el error de calibración esperado (ECE) mide si "confianza 0,9" significa acertar 9 de cada 10.
  El escalado de temperatura ajusta un solo número (T) para que lo signifique, sin cambiar qué clase se elige.
"""
import math

import numpy as np


def intervalo_bootstrap(aciertos, repeticiones=10000, nivel=0.95, semilla=0):
    """Intervalo percentil para la proporción de aciertos (lista de 0 y 1). Devuelve (media, desde, hasta)."""
    a = np.asarray(aciertos, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    gen = np.random.default_rng(semilla)
    medias = a[gen.integers(0, a.size, size=(repeticiones, a.size))].mean(axis=1)
    cola = (1 - nivel) / 2
    return float(a.mean()), float(np.quantile(medias, cola)), float(np.quantile(medias, 1 - cola))


def mcnemar(aciertos_a, aciertos_b):
    """Test de McNemar exacto (binomial, dos colas) para dos modelos sobre los mismos casos.
    Devuelve (solo_a, solo_b, p): cuántos acierta solo A, cuántos solo B y el valor p."""
    a, b = np.asarray(aciertos_a, bool), np.asarray(aciertos_b, bool)
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    n = solo_a + solo_b
    if n == 0:
        return solo_a, solo_b, 1.0
    k = min(solo_a, solo_b)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return solo_a, solo_b, min(1.0, 2 * cola)


def ece(confianzas, aciertos, bins=10):
    """Error de calibración esperado: la diferencia media entre confianza y exactitud, por franjas de confianza."""
    c, a = np.asarray(confianzas, float), np.asarray(aciertos, float)
    if c.size == 0:
        return 0.0
    franja = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum(abs(c[franja == k].mean() - a[franja == k].mean()) * (franja == k).sum()
                     for k in range(bins) if (franja == k).any()) / c.size)


def confiabilidad(confianzas, aciertos, bins=10):
    """Filas del diagrama de confiabilidad: franja, cantidad, confianza media y exactitud."""
    c, a = np.asarray(confianzas, float), np.asarray(aciertos, float)
    franja = np.minimum((c * bins).astype(int), bins - 1)
    return [{"desde": k / bins, "hasta": (k + 1) / bins, "n": int((franja == k).sum()),
             "confianza": round(float(c[franja == k].mean()), 3), "exactitud": round(float(a[franja == k].mean()), 3)}
            for k in range(bins) if (franja == k).any()]


def ajustar_temperatura(logits, etiquetas):
    """T que minimiza la log-verosimilitud negativa en validación (búsqueda dorada sobre log T, es convexa).
    logits: (n, clases) sin softmax (o log-probabilidades); etiquetas: (n,) índices."""
    z, y = np.asarray(logits, float), np.asarray(etiquetas, int)

    def nll(log_t):
        s = z / math.exp(log_t)
        s = s - s.max(axis=1, keepdims=True)
        return float(-(s[np.arange(len(y)), y] - np.log(np.exp(s).sum(axis=1))).mean())

    a, b = math.log(0.05), math.log(20.0)
    razon = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c, d = b - razon * (b - a), a + razon * (b - a)
        if nll(c) < nll(d):
            b = d
        else:
            a = c
    return math.exp((a + b) / 2)


def con_temperatura(logits, t):
    s = np.asarray(logits, float) / t
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def formato_intervalo(media, desde, hasta):
    return f"{100 * media:.0f} % [{100 * desde:.0f}–{100 * hasta:.0f}]"
