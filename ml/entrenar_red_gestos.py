"""Red de gestos, versión 2: más grande, entrenada con PyTorch, con aumento de datos y más fotos de HaGRID.

Reemplaza a ml/entrenar_mlp.py (scikit-learn), que queda para reproducir la versión 1. Lo que cambia:
- Más datos: la muestra de 254.661 fotos de HaGRID (ml/extraer_hagrid.py), 8 veces la de la versión 1.
- Red más grande: se comparan arquitecturas de 128-64 (la versión 1) hasta 1024-512-256, con dropout.
- Aumento de datos en cada época: espejo (mano izquierda ↔ derecha), rotación de la mano en el plano de la
  imagen (±25°), un poco de giro en profundidad y ruido en los puntos. Así la red ve manos inclinadas y
  detecciones imperfectas que no están en las fotos.
Lo que se mantiene: partición fija por persona (preparar.particion), elección con VALIDACIÓN y resultado en
PRUEBA (personas no vistas), pesos por clase, umbral de confianza elegido con validación, comparación con las
reglas del visor sobre las mismas filas (ml/evaluar_reglas.mjs) y exportación a JSON para el navegador.
La red v1 (frontend/ml/mlp_gestos.json antes de correr esto) también se evalúa en la misma prueba.

Uso (desde la raíz, con el entorno del backend, que tiene PyTorch):
    backend/.venv/Scripts/python ml/entrenar_red_gestos.py [--datos ml/data/externo] [--rapido]
"""
import argparse
import base64
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.stdout.reconfigure(encoding="utf-8")
ML = Path(__file__).resolve().parent
RAIZ = ML.parent
sys.path.insert(0, str(ML))
import preparar  # noqa: E402

RESULTADOS = ML / "resultados"
MODELO_WEB = RAIZ / "frontend" / "ml" / "mlp_gestos.json"
CASOS_TEST = RAIZ / "frontend" / "tests" / "datos" / "mlp_casos.json"
CLASES = sorted(preparar.CLASES)   # mismo orden que la versión 1 (scikit-learn ordena las clases)
ACCIONES = [c for c in CLASES if c != "ninguno"]

CANDIDATOS = [
    {"ocultas": [128, 64], "dropout": 0.1},
    {"ocultas": [256, 128], "dropout": 0.15},
    {"ocultas": [512, 256], "dropout": 0.2},
    {"ocultas": [512, 256, 128], "dropout": 0.2},
    {"ocultas": [1024, 512, 256], "dropout": 0.3},
]


# ---------------------------------------------------------------- aumento de datos

def aumentar(x, gen):
    """x: (n, 63) normalizado (origen en la muñeca, escala muñeca → nudillo medio). Devuelve una copia
    aumentada: espejo al azar, rotación en el plano de la imagen, giro en profundidad y ruido."""
    n = x.shape[0]
    p = x.view(n, 21, 3).clone()
    espejo = torch.rand(n, generator=gen) < 0.5
    p[espejo, :, 0] *= -1
    a = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(25)       # inclinación de la mano
    b = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(15)       # giro en profundidad (eje vertical)
    ca, sa, cb, sb = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b)
    x0, y0, z0 = p[..., 0], p[..., 1], p[..., 2]
    x1 = ca[:, None] * x0 - sa[:, None] * y0
    y1 = sa[:, None] * x0 + ca[:, None] * y0
    x2 = cb[:, None] * x1 + sb[:, None] * z0
    z2 = -sb[:, None] * x1 + cb[:, None] * z0
    p = torch.stack([x2, y1, z2], dim=-1)
    p += torch.randn(p.shape, generator=gen) * 0.02                     # temblor de la detección
    p -= p[:, :1, :].clone()                                            # la muñeca sigue en el origen
    escala = p[:, 9, :].norm(dim=1).clamp_min(1e-6)                     # y la escala sigue siendo 1
    return (p / escala[:, None, None]).reshape(n, 63)


# ---------------------------------------------------------------- red

def crear_red(ocultas, dropout):
    capas, entrada = [], 63
    for h in ocultas:
        capas += [torch.nn.Linear(entrada, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        entrada = h
    capas.append(torch.nn.Linear(entrada, len(CLASES)))
    return torch.nn.Sequential(*capas)


def probabilidades(red, X, lote=8192):
    red.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(red(X[i:i + lote]), 1) for i in range(0, len(X), lote)]).numpy()


def entrenar(cand, X, y, Xv, yv, aumento=True, epocas=60, paciencia=8, semilla=0):
    torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    red = crear_red(cand["ocultas"], cand["dropout"])
    conteo = np.bincount(y, minlength=len(CLASES))
    pesos = torch.tensor(len(y) / (len(CLASES) * np.maximum(conteo, 1)), dtype=torch.float32)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    Xvt = torch.tensor(Xv)
    mejor, estado, sin_mejora = -1, None, 0
    for epoca in range(epocas):
        red.train()
        orden = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(orden), 1024):
            idx = orden[i:i + 1024]
            xb = aumentar(Xt[idx], gen) if aumento else Xt[idx]
            opt.zero_grad()
            perdida(red(xb), yt[idx]).backward()
            opt.step()
        sched.step()
        f1 = f1_score(yv, probabilidades(red, Xvt).argmax(1), average="macro")
        if f1 > mejor + 1e-4:
            mejor, estado, sin_mejora = f1, {k: v.clone() for k, v in red.state_dict().items()}, 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(estado)
    return red, mejor, epoca + 1


def empaquetar(m):
    """float32 en base64: la mitad de tamaño que los números como texto (frontend/js/mlp.js lo lee)."""
    a = np.ascontiguousarray(m, dtype="<f4")
    return {"forma": list(a.shape), "float32": base64.b64encode(a.tobytes()).decode("ascii")}


def capas_de(red):
    lineales = [m for m in red if isinstance(m, torch.nn.Linear)]
    return [(m.weight.detach().numpy().T, m.bias.detach().numpy()) for m in lineales]


# ---------------------------------------------------------------- evaluación

def con_umbral(proba, umbral):
    k = proba.argmax(1)
    return np.where(proba.max(1) >= umbral, k, CLASES.index("ninguno"))


def elegir_umbral(proba, y):
    mejor, umbral = -1, 0.0
    for u in np.arange(0.3, 0.96, 0.05):
        f1 = f1_score(y, con_umbral(proba, u), average="macro")
        if f1 > mejor + 1e-4:
            mejor, umbral = f1, float(round(u, 2))
    return umbral, mejor


def resumen(nombre, y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    ninguno = y == "ninguno"
    return {
        "metodo": nombre,
        "exactitud": round(float(np.mean(y == pred)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        "recall_gestos": round(float(np.mean([np.mean(pred[y == c] == c) for c in ACCIONES])), 4),
        "falsas_alarmas_ninguno": round(float(np.mean(pred[ninguno] != "ninguno")), 4),
    }


def sesgos(df_prueba, y, pred):
    """Recall de los gestos con cada mano y por persona (personas con al menos 3 gestos)."""
    y, pred = np.asarray(y), np.asarray(pred)
    gesto = y != "ninguno"
    por_mano = {m: round(float(np.mean(pred[gesto & (df_prueba["mano"].to_numpy() == m)]
                                     == y[gesto & (df_prueba["mano"].to_numpy() == m)])), 4)
                for m in ("derecha", "izquierda")}
    personas = df_prueba["persona"].to_numpy()
    aciertos = {}
    for p in np.unique(personas[gesto]):
        sel = gesto & (personas == p)
        if sel.sum() >= 3:
            aciertos[p] = float(np.mean(pred[sel] == y[sel]))
    valores = np.array(list(aciertos.values()))
    return {"recall_por_mano": por_mano, "personas_con_3_gestos": int(len(valores)),
            "mediana_por_persona": round(float(np.median(valores)), 4),
            "personas_bajo_80": int(np.sum(valores < 0.8))}


def predecir_json(modelo, X):
    a = X
    for k, c in enumerate(modelo["capas"]):
        a = a @ np.asarray(c["W"], dtype=np.float32) + np.asarray(c["b"], dtype=np.float32)
        if k < len(modelo["capas"]) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    k = p.argmax(1)
    clases = np.array(modelo["clases"])
    return np.where(p.max(1) >= modelo["umbral_confianza"], clases[k], "ninguno")


def reglas_en(df_prueba):
    """Predicciones de las reglas del visor (gestures.js, con node) sobre las filas de prueba."""
    carpeta = ML / "data" / "externo" / "prueba"
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*.csv"):
        viejo.unlink()
    columnas = [c for c in df_prueba.columns if c not in ("archivo", "fila", "particion")]
    df_prueba[columnas].to_csv(carpeta / "hagrid_prueba.csv", index=False)
    salida = RESULTADOS / "hagrid_reglas_prueba.csv"
    subprocess.run(["node", str(ML / "evaluar_reglas.mjs"), str(carpeta), str(salida)], check=True, cwd=RAIZ)
    import pandas as pd
    return pd.read_csv(salida)["prediccion"].to_numpy()


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=ML / "data" / "externo")
    ap.add_argument("--rapido", action="store_true", help="pocas épocas y dos candidatos (para probar el script)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    df = preparar.cargar(args.datos)
    df["particion"] = df["persona"].map(preparar.particion)
    X = preparar.normalizar(df)
    y = np.array([CLASES.index(c) for c in df["clase"]])
    parte = df["particion"].to_numpy()
    ent, val, pru = parte == "entrenamiento", parte == "validacion", parte == "prueba"
    print(f"manos: entrenamiento {ent.sum()} · validación {val.sum()} · prueba {pru.sum()} · "
          f"personas {df['persona'].nunique()}")

    candidatos = CANDIDATOS[:2] if args.rapido else CANDIDATOS
    epocas = 4 if args.rapido else 60

    # 1) Arquitectura, con aumento de datos, elegida con validación
    tabla, elegido = [], None
    for cand in candidatos:
        t0 = time.perf_counter()
        red, f1, n_ep = entrenar(cand, X[ent], y[ent], X[val], y[val], epocas=epocas)
        params = sum(p.numel() for p in red.parameters())
        fila = {"ocultas": cand["ocultas"], "dropout": cand["dropout"], "parametros": params,
                "f1_validacion": round(f1, 4), "epocas": n_ep, "segundos": round(time.perf_counter() - t0)}
        tabla.append(fila)
        print(f"  {str(cand['ocultas']):18s} {params:>8,} parámetros · F1 validación {f1:.4f} · "
              f"{n_ep} épocas · {fila['segundos']} s")
        # Una red más grande tiene que ganar por algo (0,002 de F1) para justificar su tamaño
        if elegido is None or f1 > elegido[1] + 0.002:
            elegido = (cand, f1, red)
    cand, f1_elegido, red = elegido

    # 2) ¿Cuánto aporta el aumento de datos? La misma arquitectura sin aumento
    red_sin, f1_sin, _ = entrenar(cand, X[ent], y[ent], X[val], y[val], aumento=False, epocas=epocas)
    print(f"elegida {cand['ocultas']} · F1 validación con aumento {f1_elegido:.4f} · sin aumento {f1_sin:.4f}")

    # 3) Umbral de confianza con validación
    umbral, f1_u = elegir_umbral(probabilidades(red, torch.tensor(X[val])), y[val])
    print(f"umbral de confianza {umbral} (F1 validación {f1_u:.4f})")

    # 4) Prueba: reglas, red v1 (la que estaba en el visor) y red v2, sobre las mismas manos
    y_pru = np.array(CLASES)[y[pru]]
    pred_v2 = np.array(CLASES)[con_umbral(probabilidades(red, torch.tensor(X[pru])), umbral)]
    comparacion = []
    df_pru = df[pru].reset_index(drop=True)
    reglas = reglas_en(df_pru)
    comparacion.append(resumen("reglas", y_pru, reglas))
    if MODELO_WEB.exists():
        v1 = json.loads(MODELO_WEB.read_text(encoding="utf-8"))
        if isinstance(v1["capas"][0]["W"], list):   # la v1 guarda los pesos como listas
            comparacion.append(resumen(f"mlp v1 {[len(c['b']) for c in v1['capas'][:-1]]}", y_pru,
                                       predecir_json(v1, X[pru])))
    comparacion.append(resumen(f"red v2 {cand['ocultas']}", y_pru, pred_v2))
    import pandas as pd
    print("\n" + pd.DataFrame(comparacion).to_string(index=False))
    print("\nRed v2 por clase (prueba):\n" + classification_report(y_pru, pred_v2, digits=3, zero_division=0))
    sesgo = sesgos(df_pru, y_pru, pred_v2)
    print(f"sesgos: {sesgo}")

    if args.rapido:
        print("(--rapido: no se exporta nada)")
        return

    # 5) Reporte y exportación para el navegador
    RESULTADOS.mkdir(exist_ok=True)
    pd.DataFrame(confusion_matrix(y_pru, pred_v2, labels=CLASES), index=CLASES, columns=CLASES) \
        .to_csv(RESULTADOS / "hagrid_red_v2_confusion.csv", encoding="utf-8")
    reporte = {
        "fecha": date.today().isoformat(),
        "dataset": f"HaGRID ({args.datos.relative_to(RAIZ) if args.datos.is_relative_to(RAIZ) else args.datos}), "
                   "CC BY-SA 4.0",
        "filas": {"entrenamiento": int(ent.sum()), "validacion": int(val.sum()), "prueba": int(pru.sum())},
        "personas": {k: int(df.loc[parte == k, "persona"].nunique()) for k in ("entrenamiento", "validacion", "prueba")},
        "arquitecturas_validacion": tabla,
        "elegida": {"ocultas": cand["ocultas"], "dropout": cand["dropout"]},
        "aumento_de_datos": {"f1_validacion_con": round(f1_elegido, 4), "f1_validacion_sin": round(f1_sin, 4)},
        "umbral_confianza": umbral,
        "comparacion_prueba": comparacion,
        "sesgos_prueba": sesgo,
        "por_clase": classification_report(y_pru, pred_v2, output_dict=True, zero_division=0),
    }
    (RESULTADOS / "hagrid_red_v2_reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")

    capas = capas_de(red)
    web = {
        "clases": CLASES,
        "umbral_confianza": umbral,
        "capas": [{"W": empaquetar(W), "b": empaquetar(b)} for W, b in capas],
        "activacion": "relu",
        "entrada": "63 valores normalizados (frontend/js/normalize.js: aVector(normalizar(puntos, aspecto)))",
        "origen": reporte["dataset"],
        "version": 2,
        "fecha": reporte["fecha"],
    }
    MODELO_WEB.write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rng = np.random.default_rng(0)
    muestras = rng.choice(np.flatnonzero(pru), 12, replace=False)
    proba = probabilidades(red, torch.tensor(X[muestras]))
    casos = [{"x": np.round(X[i], 6).tolist(), "proba": np.round(p, 5).tolist()} for i, p in zip(muestras, proba)]
    CASOS_TEST.write_text(json.dumps({"clases": CLASES, "casos": casos}), encoding="utf-8")
    print(f"\nModelo para el navegador: {MODELO_WEB.relative_to(RAIZ)} ({MODELO_WEB.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
