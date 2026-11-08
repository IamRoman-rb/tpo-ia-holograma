"""Reconocimiento de piezas de motores por su forma: entrena la red y la exporta para el navegador.

Los modelos de Objaverse se desarman en el visor, pero sus piezas se llaman por su lugar ("Sección superior
izquierda"). Esta red mira la forma de cada pieza y su lugar en el conjunto (ml/piezas/rasgos.py) y dice qué
pieza de motor es: pistón, biela, cigüeñal, culata, rotor, estator…

Datos: motores generados por código (backend/motores/), que traen cada pieza con su clase. Se generan miles de
variantes (cilindros, disposición, medidas, válvulas, turbo, diésel, eléctricos de distintos tipos y potencias) y
a cada una se le aplica una deformación al azar: giro, estiramiento por eje y piezas que faltan, porque un modelo
real nunca es igual a uno generado. Se agregan además piezas que no son de motor (cajas, cilindros, placas, bases,
piezas de otros objetos) con la clase "otro": así la red aprende a decir "no sé" en vez de forzar un nombre.

Evaluación honesta: las familias de motores de la prueba (disposición + cantidad de cilindros) no aparecen en el
entrenamiento, así se mide si reconoce piezas de motores que nunca vio. No hay modelos reales etiquetados: el
resultado con los de Objaverse se revisa a ojo y queda anotado en el reporte como pendiente de medir.

Red: PointNet chica (Qi et al., 2017). Una misma red de 3 capas mira cada punto de la pieza, se toma el máximo y
el promedio sobre los puntos (no importa el orden de los puntos) y se le suma el contexto de la pieza. Es chica
a propósito: corre en el navegador, sin librerías, para todas las piezas de un modelo en menos de un segundo.

Uso (con el entorno de entrenamiento, desde la raíz):
    backend/.venv/Scripts/python ml/piezas/entrenar_piezas.py            # ~20 min en CPU
    backend/.venv/Scripts/python ml/piezas/entrenar_piezas.py --rapido   # ~2 min, no exporta
Salida: frontend/ml/piezas_motor.json (pesos, clases, nombres, descripciones, normalización, umbral),
ml/resultados/piezas_reporte.json y frontend/tests/datos/piezas_fixture.json (para comparar Python y JS).
"""
import argparse
import base64
import json
import math
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from geometria import caja, cilindro, cono, esfera, mover, rot_x, rot_y, rot_z, toroide  # noqa: E402
from motores.combustion import Combustion  # noqa: E402
from motores.combustion import generar as generar_combustion  # noqa: E402
from motores.electrico import Electrico  # noqa: E402
from motores.electrico import generar as generar_electrico  # noqa: E402
from rasgos import CONTEXTO, muestrear, rasgos_de_muestras  # noqa: E402

PUNTOS = 256                      # por pieza: alcanza para la forma y es la mitad de cuentas en el navegador
# Confianza mínima para nombrar una pieza. En validación (motores generados) alcanza con mucho menos, pero con
# modelos reales de Objaverse la red se equivoca mucho más (sus piezas son trozos arbitrarios del archivo, no
# piezas limpias): por debajo de 0,8 nombraba 19 bielas en un V8. Ver "modelos_reales" en el reporte.
UMBRAL_MINIMO = 0.8
OTRO = "otro"
SALIDA = RAIZ / "frontend" / "ml" / "piezas_motor.json"
REPORTE = RAIZ / "ml" / "resultados" / "piezas_reporte.json"
FIXTURE = RAIZ / "frontend" / "tests" / "datos" / "piezas_fixture.json"

# Familias de la prueba: nunca se entrenan. Una de cada disposición y un rango de potencias de los eléctricos.
FAMILIAS_PRUEBA = {("linea", 5), ("v", 10), ("boxer", 2), ("radial", 5)}
POTENCIA_PRUEBA = (1.5, 3.0)      # kW


# ---------------------------------------------------------------- motores al azar

def combustion_al_azar(rng):
    disposicion = rng.choice(["linea", "linea", "v", "boxer", "radial"])
    cilindros = int(rng.choice({"linea": [1, 2, 3, 4, 5, 6], "v": [6, 8, 10, 12], "boxer": [2, 4, 6],
                                "radial": [5, 7, 9]}[disposicion]))
    diametro = float(rng.uniform(6.0, 12.0))
    return Combustion(
        cilindros=cilindros, disposicion=disposicion, angulo_v=float(rng.choice([60.0, 72.0, 90.0])),
        diametro=round(diametro, 1), carrera=round(diametro * float(rng.uniform(0.8, 1.25)), 1),
        arboles=str(rng.choice(["sohc", "dohc"])), valvulas=int(rng.choice([2, 4])), turbo=bool(rng.random() < 0.4),
        distribucion=str(rng.choice(["correa", "cadena"])), combustible=str(rng.choice(["nafta", "diesel"])))


def electrico_al_azar(rng):
    tipo = str(rng.choice(["induccion", "induccion", "continua", "brushless"]))
    return Electrico(tipo, round(float(np.exp(rng.uniform(np.log(0.1), np.log(50.0)))), 2),
                     str(rng.choice(["patas", "brida"])) if tipo == "induccion" else "patas")


def es_de_prueba(params):
    if isinstance(params, Electrico):
        return POTENCIA_PRUEBA[0] <= params.potencia_kw <= POTENCIA_PRUEBA[1]
    return (params.disposicion, params.cilindros) in FAMILIAS_PRUEBA


def piezas_del_modelo(modelo):
    """[(clase, nombre, descripcion, posiciones, triángulos)] de un Modelo generado."""
    salida = []
    for pieza in modelo.piezas.values():
        partes = [p for lista in pieza["partes"].values() for p in lista]
        if not partes:
            continue
        P, I, base = [], [], 0
        for pos, _, idx in partes:
            P.append(np.asarray(pos, np.float64))
            I.append(np.asarray(idx).reshape(-1, 3) + base)
            base += len(pos)
        meta = pieza["meta"]
        salida.append((meta["clase"], meta["nombre"], meta["descripcion"], np.concatenate(P), np.concatenate(I)))
    return salida


# ---------------------------------------------------------------- piezas que no son de motor ("otro")

def distractor(rng, tam):
    """Una pieza cualquiera (caja, placa, base, esfera, aro, cono, cilindro) del orden de tamaño del motor."""
    s = tam * float(rng.uniform(0.08, 0.6))
    tipo = rng.integers(7)
    if tipo == 0:
        parte = caja(*(s * rng.uniform(0.3, 1.0, 3)))
    elif tipo == 1:                                   # placa o base: chata y ancha
        parte = caja(s * 1.5, s * 0.05, s * float(rng.uniform(0.8, 1.5)))
    elif tipo == 2:
        parte = esfera(s / 2, 16, 8)
    elif tipo == 3:
        parte = toroide(s / 2, s * float(rng.uniform(0.05, 0.2)), 32, 8)
    elif tipo == 4:
        parte = cono(s / 2, s * float(rng.uniform(0.0, 0.3)), s, 24)
    elif tipo == 5:
        parte = cilindro(s * float(rng.uniform(0.05, 0.5)), s * float(rng.uniform(0.5, 2.0)), 24)
    else:                                             # pie o soporte: una columna con su base
        a = cilindro(s * 0.08, s, 16)
        b = mover(caja(s * 0.6, s * 0.05, s * 0.6), (0, -s / 2, 0))
        parte = (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]),
                 np.concatenate([np.asarray(a[2], np.int64), np.asarray(b[2], np.int64) + len(a[0])]))
    R = rot_z(float(rng.uniform(0, 2 * math.pi))) @ rot_y(float(rng.uniform(0, 2 * math.pi))) @ rot_x(
        float(rng.uniform(0, 2 * math.pi)))
    pos, _, idx = mover(parte, R=R)
    return np.asarray(pos, np.float64), np.asarray(idx).reshape(-1, 3)


# ---------------------------------------------------------------- de motor a ejemplos

def deformar(piezas, rng):
    """Lo que hace distinto a un modelo real de uno generado: giro, estiramiento por eje, piezas que faltan y
    piezas que no son del motor (pie, base, placa con el nombre…)."""
    piezas = list(piezas)
    # Faltan piezas: cada una con probabilidad hasta 30 % (quedan por lo menos 4)
    falta = float(rng.uniform(0, 0.3))
    quedan = [p for p in piezas if rng.random() > falta]
    piezas = quedan if len(quedan) >= 4 else piezas
    todos = np.concatenate([p[3] for p in piezas])
    centro, tam = todos.mean(0), float(np.ptp(todos, 0).max())
    # Piezas que no son del motor, alrededor de él
    for _ in range(int(rng.choice([0, 0, 1, 2, 3]))):
        pos, idx = distractor(rng, tam)
        pos = pos + centro + rng.normal(0, tam * 0.35, 3)
        piezas.append((OTRO, "", "", pos, idx))
    # Giro al azar y estiramiento por eje (±15 %)
    R = rot_z(float(rng.uniform(0, 2 * math.pi))) @ rot_y(float(rng.uniform(0, 2 * math.pi))) @ rot_x(
        float(rng.uniform(0, 2 * math.pi)))
    S = np.diag(rng.uniform(0.85, 1.15, 3))
    M = R @ S
    return [(c, n, d, (pos - centro) @ M.T, idx) for c, n, d, pos, idx in piezas]


def ejemplos(params, rng, copias):
    modelo = generar_electrico(params) if isinstance(params, Electrico) else generar_combustion(params)
    base = piezas_del_modelo(modelo)
    for _ in range(copias):
        piezas = deformar(base, rng)
        muestras, areas = zip(*(muestrear(pos, idx, PUNTOS, rng) for _, _, _, pos, idx in piezas))
        puntos, contexto = rasgos_de_muestras(muestras, areas)
        # Un poco de ruido en los puntos: las mallas reales no son perfectas
        puntos = puntos + rng.normal(0, 0.01, puntos.shape).astype(np.float32)
        yield puntos, contexto, [p[0] for p in piezas]


def generar_datos(n_modelos, copias, semilla):
    rng = np.random.default_rng(semilla)
    grupos = {"entrenamiento": [], "prueba": []}
    textos = {}                                       # clase -> (nombre sin número, descripción, familia)
    maximos = Counter()                               # clase -> cuántas puede tener un motor, como mucho
    vistos = set()
    while sum(len(v) for v in grupos.values()) < n_modelos:
        params = electrico_al_azar(rng) if rng.random() < 0.25 else combustion_al_azar(rng)
        clave = repr(params)
        if clave in vistos:
            continue
        vistos.add(clave)
        try:
            modelo = generar_electrico(params) if isinstance(params, Electrico) else generar_combustion(params)
        except Exception as e:                         # alguna combinación que el generador no admite
            print(f"  salteo {params}: {e}")
            continue
        familia = "electrico" if isinstance(params, Electrico) else "combustion"
        for clase, nombre, desc, _, _ in piezas_del_modelo(modelo):
            textos.setdefault(clase, (nombre, desc, set()))[2].add(familia)
        for clase, n in Counter(p[0] for p in piezas_del_modelo(modelo)).items():
            maximos[clase] = max(maximos[clase], n)
        grupos["prueba" if es_de_prueba(params) else "entrenamiento"].append(params)

    datos = {}
    for parte, lista in grupos.items():
        P, C, Y, M = [], [], [], []
        for k, params in enumerate(lista):
            for puntos, contexto, clases in ejemplos(params, rng, copias):
                P.append(puntos)
                C.append(contexto)
                Y.extend(clases)
                M.extend([k] * len(clases))
        datos[parte] = (np.concatenate(P), np.concatenate(C), np.array(Y), np.array(M))
        print(f"  {parte}: {len(lista)} motores, {len(datos[parte][2])} piezas")
    return datos, textos, maximos


# ---------------------------------------------------------------- nombres para mostrar

def nombre_de_clase(clase, ejemplo):
    """"Pistón 3" → "Pistón"; "Tapa de válvulas derecho" → "Tapa de válvulas"."""
    n = ejemplo.split(" (")[0]
    for sufijo in (" izquierdo", " derecho", " delantero", " trasero"):
        if n.endswith(sufijo):
            n = n[: -len(sufijo)]
    partes = n.split()
    while partes and partes[-1].isdigit():
        partes.pop()
    n = " ".join(partes) or clase.replace("_", " ").capitalize()
    return n


def sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- red

def entrenar(datos, clases, epocas, semilla):
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    indice = {c: i for i, c in enumerate(clases)}
    Ptr, Ctr, Ytr, Mtr = datos["entrenamiento"]
    # Validación: el 15 % de los motores de entrenamiento (enteros, no piezas sueltas)
    rng = np.random.default_rng(semilla)
    motores_val = set(rng.choice(np.unique(Mtr), max(1, len(np.unique(Mtr)) * 15 // 100), replace=False))
    es_val = np.isin(Mtr, list(motores_val))
    media, desvio = Ctr[~es_val].mean(0), Ctr[~es_val].std(0) + 1e-6

    def tensores(P, C, Y):
        return (torch.from_numpy(P), torch.from_numpy(((C - media) / desvio).astype(np.float32)),
                torch.tensor([indice[y] for y in Y]))

    tr = tensores(Ptr[~es_val], Ctr[~es_val], Ytr[~es_val])
    va = tensores(Ptr[es_val], Ctr[es_val], Ytr[es_val])

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.punto = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128),
                                       nn.ReLU())
            self.cabeza = nn.Sequential(nn.Linear(256 + CONTEXTO, 128), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(clases)))

        def forward(self, p, c):
            f = self.punto(p)
            return self.cabeza(torch.cat([f.max(1).values, f.mean(1), c], 1))

    red = Red()
    # Pesos por clase (raíz de la frecuencia inversa): las piezas únicas (base, chaveta…) no quedan tapadas
    cuenta = np.bincount(tr[2].numpy(), minlength=len(clases)).astype(np.float64)
    pesos = torch.tensor(np.where(cuenta > 0, 1 / np.sqrt(cuenta + 1), 0) * len(clases) / np.sum(
        np.where(cuenta > 0, 1 / np.sqrt(cuenta + 1), 0)), dtype=torch.float32)
    perdida = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-4)
    pasos_por_epoca = math.ceil(len(tr[2]) / 256)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epocas * pasos_por_epoca)

    def evaluar(t):
        red.eval()
        with torch.no_grad():
            prob = torch.cat([torch.softmax(red(t[0][i:i + 2048], t[1][i:i + 2048]), 1)
                              for i in range(0, len(t[2]), 2048)])
        red.train()
        return prob.numpy()

    mejor, estado = -1.0, None
    for e in range(epocas):
        orden = torch.randperm(len(tr[2]))
        t0, total = time.time(), 0.0
        for k in range(pasos_por_epoca):
            i = orden[k * 256:(k + 1) * 256]
            # Aumento en el momento: subconjunto de puntos repetido y un giro chico (la pieza ya viene en los ejes
            # del conjunto, pero esos ejes en un modelo real no son exactos)
            p = tr[0][i]
            ang = (torch.rand(3) - 0.5) * 0.3
            cx, cy, cz = torch.cos(ang)
            sx, sy, sz = torch.sin(ang)
            Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            p = p @ (Rz @ Ry @ Rx).T
            loss = perdida(red(p, tr[1][i]), tr[2][i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            plan.step()
            total += loss.item()
        acierto = float((evaluar(va).argmax(1) == va[2].numpy()).mean())
        print(f"  época {e + 1}/{epocas}: pérdida {total / pasos_por_epoca:.3f} · validación {acierto:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if acierto >= mejor:
            mejor, estado = acierto, {k: v.clone() for k, v in red.state_dict().items()}
    red.load_state_dict(estado)
    return red, media, desvio, va, evaluar


def umbral_por_precision(prob, y, otro, meta=0.95):
    """La confianza mínima con la que, en validación, los nombres que se ponen aciertan por lo menos meta. Entre
    los umbrales que la cumplen, el que nombra más piezas. Las piezas "otro" que se nombran cuentan como error."""
    pred, conf = prob.argmax(1), prob.max(1)
    mejor = 0.99
    for u in np.arange(0.30, 0.99, 0.01):
        nombra = (conf >= u) & (pred != otro)
        if nombra.sum() == 0:
            continue
        if (pred[nombra] == y[nombra]).mean() >= meta:
            mejor = float(u)
            break
    return round(mejor, 2)


# ---------------------------------------------------------------- exportar

def b64(t):
    a = np.ascontiguousarray(t.detach().numpy() if hasattr(t, "detach") else t, np.float32)
    return {"forma": list(a.shape), "float32": base64.b64encode(a.tobytes()).decode()}


def exportar(red, clases, textos, maximos, media, desvio, umbral):
    lineales = [m for m in list(red.punto) + list(red.cabeza) if hasattr(m, "weight")]
    capas = [{"W": b64(m.weight.T), "b": b64(m.bias)} for m in lineales]
    info = {}
    for c in clases:
        if c == OTRO:
            info[c] = {"nombre": "", "descripcion": "", "familias": []}
            continue
        nombre, desc, familias = textos[c]
        info[c] = {"nombre": nombre_de_clase(c, nombre), "descripcion": desc, "familias": sorted(familias),
                   "maximo": int(maximos[c])}
    modelo = {
        "version": 1, "puntos": PUNTOS, "contexto": CONTEXTO, "clases": clases, "info": info, "otro": OTRO,
        "umbral_confianza": umbral, "media_contexto": b64(media), "desvio_contexto": b64(desvio),
        # capas 0-2: red de cada punto (ReLU); se toma máximo y promedio; capas 3-5: cabeza (ReLU, ReLU, lineal)
        "capas_punto": capas[:3], "capas_cabeza": capas[3:],
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(modelo, ensure_ascii=False), encoding="utf-8")
    print(f"  exportado {SALIDA.relative_to(RAIZ)} ({SALIDA.stat().st_size // 1024} KB)")


def exportar_fixture(red, media, desvio, semilla):
    """Un motor chico con sus puntos ya muestreados, y lo que Python calcula: los rasgos y las probabilidades.
    frontend/tests/reconocer_piezas.test.mjs verifica que el navegador da lo mismo."""
    import torch
    rng = np.random.default_rng(semilla)
    piezas = piezas_del_modelo(generar_combustion(Combustion(cilindros=2, valvulas=2, arboles="sohc")))[:8]
    muestras, areas = zip(*(muestrear(pos, idx, PUNTOS, rng) for _, _, _, pos, idx in piezas))
    puntos, contexto = rasgos_de_muestras(muestras, areas)
    red.eval()
    with torch.no_grad():
        prob = torch.softmax(red(torch.from_numpy(puntos),
                                 torch.from_numpy(((contexto - media) / desvio).astype(np.float32))), 1).numpy()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({
        "muestras": [np.round(m, 6).tolist() for m in muestras], "areas": [float(a) for a in areas],
        "puntos": np.round(puntos, 6).tolist(), "contexto": np.round(contexto, 6).tolist(),
        "probabilidades": np.round(prob, 6).tolist(), "clases_reales": [p[0] for p in piezas],
    }), encoding="utf-8")
    print(f"  fixture {FIXTURE.relative_to(RAIZ)}")


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelos", type=int, default=1200, help="motores distintos a generar")
    ap.add_argument("--copias", type=int, default=3, help="deformaciones de cada motor")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--rapido", action="store_true", help="versión mínima para probar que todo anda; no exporta")
    a = ap.parse_args()
    if a.rapido:
        a.modelos, a.copias, a.epocas = 80, 1, 3

    t0 = time.time()
    print("Generando motores…", flush=True)
    datos, textos, maximos = generar_datos(a.modelos, a.copias, a.semilla)
    clases = sorted(textos) + [OTRO]
    print(f"  {len(clases)} clases ({time.time() - t0:.0f} s)")

    print("Entrenando…", flush=True)
    red, media, desvio, va, evaluar = entrenar(datos, clases, a.epocas, a.semilla)
    import torch
    indice = {c: i for i, c in enumerate(clases)}
    otro = indice[OTRO]
    prob_va = evaluar(va)
    umbral_validacion = umbral_por_precision(prob_va, va[2].numpy(), otro)
    umbral = max(umbral_validacion, UMBRAL_MINIMO)

    # Prueba: familias de motores que la red nunca vio
    Pte, Cte, Yte, _ = datos["prueba"]
    te = (torch.from_numpy(Pte), torch.from_numpy(((Cte - media) / desvio).astype(np.float32)),
          torch.tensor([indice[y] for y in Yte]))
    prob = evaluar(te)
    y, pred, conf = te[2].numpy(), prob.argmax(1), prob.max(1)
    nombra = (conf >= umbral) & (pred != otro)
    por_clase = {}
    for c in clases:
        k = y == indice[c]
        if k.any():
            por_clase[c] = {"piezas": int(k.sum()), "acierto": round(float((pred[k] == y[k]).mean()), 3),
                            "nombradas": round(float(nombra[k].mean()), 3)}
    confusiones = Counter((clases[a_], clases[b]) for a_, b in zip(y, pred) if a_ != b)
    reporte = {
        "fecha": time.strftime("%Y-%m-%d"), "motores": a.modelos, "copias": a.copias, "epocas": a.epocas,
        "puntos_por_pieza": PUNTOS, "clases": len(clases),
        "piezas_entrenamiento": int(len(datos["entrenamiento"][2])), "piezas_prueba": int(len(y)),
        "familias_prueba": sorted(f"{d} {c}" for d, c in FAMILIAS_PRUEBA) + [
            f"eléctricos de {POTENCIA_PRUEBA[0]} a {POTENCIA_PRUEBA[1]} kW"],
        "validacion_acierto": round(float((prob_va.argmax(1) == va[2].numpy()).mean()), 3),
        "umbral_confianza": umbral, "umbral_validacion": umbral_validacion,
        "prueba_acierto": round(float((pred == y).mean()), 3),
        "prueba_nombradas": round(float(nombra.mean()), 3),
        "prueba_acierto_nombradas": round(float((pred[nombra] == y[nombra]).mean()), 3) if nombra.any() else None,
        "prueba_otro_nombradas_como_pieza": round(float(nombra[y == otro].mean()), 3) if (y == otro).any() else None,
        "por_clase": por_clase,
        "confusiones_frecuentes": [{"real": r, "predicha": p, "veces": n} for (r, p), n in confusiones.most_common(15)],
        "modelos_reales": "pendiente: no hay modelos de Objaverse etiquetados; se revisan a ojo en el visor",
        "segundos": round(time.time() - t0),
    }
    print(json.dumps({k: v for k, v in reporte.items() if k not in ("por_clase", "confusiones_frecuentes")},
                     ensure_ascii=False, indent=1))
    if a.rapido:
        return
    exportar(red, clases, textos, maximos, media, desvio, umbral)
    exportar_fixture(red, media, desvio, a.semilla)
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    REPORTE.write_text(json.dumps(reporte, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  reporte {REPORTE.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
