"""Deep learning para los pedidos: ajuste completo de un transformer (motor "transformer" del backend).

Se ajustan TODOS los pesos de un codificador preentrenado con dos cabezas a la vez (backend/intent_transformer.py):
la acción (sobre el primer token) y el objeto como etiquetado BIO por palabra, decodificado con Viterbi.
Candidatos, del más chico al más grande (se elige con validación; a igual resultado gana el más grande):
  - BETO (dccuchile/bert-base-spanish-wwm-cased, 110 M): BERT entrenado solo con español;
  - mDeBERTa-v3 (microsoft/mdeberta-v3-base, 280 M): atención desenredada, multilingüe;
  - XLM-RoBERTa large (FacebookAI/xlm-roberta-large, 560 M): RoBERTa multilingüe, 100 idiomas.
(El RoBERTa large del Plan TL en español ya no publica sus pesos en Hugging Face.)
Datos: una muestra balanceada de ml/intencion/frases_entrenamiento.csv (todas las acciones, incluidos los
comandos de los modelos), 2 épocas, AdamW con calentamiento y caída lineal, pérdida = entropía de la acción +
entropía BIO. Al final se calibra una temperatura con validación (eval/estadistica.py).
Validación: frases_validacion.csv, frases_objetos_validacion.csv (objeto exacto), frases_comandos_validacion.csv
y 300 frases generadas con objetos reservados. La prueba se mide con eval/eval_intent.py --motor transformer.

Uso (desde la raíz, con el entorno del backend):
    backend/.venv/Scripts/python ml/intencion_transformer/entrenar_transformer.py [--candidatos beto,mdeberta,xlmr]
Salida: backend/modelos/transformer/ (no se versiona) y ml/resultados/transformer_reporte.json
"""
import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))
import intent_modelo  # noqa: E402
import intent_transformer as it  # noqa: E402
from estadistica import ajustar_temperatura, con_temperatura, ece  # noqa: E402

SALIDA = it.RUTA_MODELO
TEMPORAL = RAIZ / "backend" / "modelos" / "transformer_candidatos"
REPORTE = RAIZ / "ml" / "resultados" / "transformer_reporte.json"
CANDIDATOS = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "xlmr": "FacebookAI/xlm-roberta-large",
}
POR_ACCION = {"mostrar": 2600, "explotar": 1300, "aislar": 1300}   # el resto: 500 cada una
OTRAS = 500
EPOCAS = 2
LOTE = 16
TASAS = {"beto": 5e-5, "mdeberta": 4e-5, "xlmr": 2e-5}   # los modelos grandes se ajustan con tasas más bajas


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def muestra(filas, r):
    por_accion = {}
    for f in filas:
        por_accion.setdefault(f["accion"], []).append(f)
    elegidas = []
    for accion, lista in sorted(por_accion.items()):
        r.shuffle(lista)
        elegidas += lista[:POR_ACCION.get(accion, OTRAS)]
    r.shuffle(elegidas)
    return elegidas


def etiquetas_bio(frase, objeto):
    """O/B/I por palabra (las palabras de intent_modelo), marcando la primera aparición del objeto."""
    palabras = intent_modelo.palabras(frase)
    objetivo = intent_modelo.palabras(objeto) if objeto else []
    marcas = [0] * len(palabras)
    n = len(objetivo)
    for i in range(len(palabras) - n + 1):
        if n and palabras[i:i + n] == objetivo:
            marcas[i] = 1
            marcas[i + 1:i + n] = [2] * (n - 1)
            break
    return marcas


def lote(tok, filas, acciones):
    """Tokeniza y alinea: la etiqueta BIO de cada palabra va en su primera sub-palabra; el resto, -100."""
    x = tok([f["frase"] for f in filas], padding=True, truncation=True, max_length=64, return_tensors="pt",
            return_offsets_mapping=True)
    desplazamientos = x.pop("offset_mapping").tolist()
    bio = torch.full(x["input_ids"].shape, -100, dtype=torch.long)
    for i, f in enumerate(filas):
        marcas = etiquetas_bio(f["frase"], f.get("objeto") or "")
        for marca, k in zip(marcas, it.posiciones_palabras(f["frase"], desplazamientos[i])):
            if k is not None:
                bio[i, k] = marca
    accion = torch.tensor([acciones.index(f["accion"]) for f in filas])
    return x["input_ids"], x["attention_mask"], accion, bio


def evaluar(interprete, conjuntos):
    """Exactitud de la acción y, en los sets con objeto, de acción + objeto exacto. También la confianza."""
    resultados, logits, etiquetas = {}, [], []
    for nombre, (filas, con_objeto) in conjuntos.items():
        acc = completo = 0
        for f in filas:
            p, log_prob = interprete.analizar(f["frase"])
            a = interprete.acciones[int(np.argmax(p))]
            logits.append(np.log(np.maximum(p, 1e-12)) * interprete.temperatura)
            etiquetas.append(interprete.acciones.index(f["accion"]))
            acc += a == f["accion"]
            if con_objeto:
                objeto = interprete.objeto(f["frase"], log_prob) if a in intent_modelo.ACCIONES_CON_OBJETO else None
                completo += a == f["accion"] and intent_modelo.palabras(objeto or "") == intent_modelo.palabras(f["objeto"])
        n = len(filas)
        resultados[nombre] = {"frases": n, "accion": round(acc / n, 4),
                              "accion_y_objeto_exacto": round(completo / n, 4) if con_objeto else None}
    return resultados, np.array(logits), np.array(etiquetas)


def puntaje(resultados):
    """Promedio de lo que importa: la acción en todos los sets y el objeto exacto donde se mide."""
    valores = [r["accion_y_objeto_exacto"] if r["accion_y_objeto_exacto"] is not None else r["accion"]
               for r in resultados.values()]
    return float(np.mean(valores))


def entrenar(nombre, base, entrenamiento, acciones, rapido=False, epocas=EPOCAS):
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(base)
    red = it.crear_red(base, len(acciones))
    parametros = sum(p.numel() for p in red.parameters())
    pasos_epoca = math.ceil(len(entrenamiento) / LOTE)
    pasos = 20 if rapido else int(epocas * pasos_epoca)
    opt = torch.optim.AdamW(red.parameters(), lr=TASAS[nombre], weight_decay=0.01)
    plan = get_linear_schedule_with_warmup(opt, max(1, pasos // 10), pasos)
    perdida_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    print(f"\n== {nombre} ({base}): {parametros / 1e6:.0f} M parámetros · {pasos} pasos de {LOTE}", flush=True)
    red.train()
    t0, perdidas, r = time.perf_counter(), [], random.Random(0)
    for k in range(pasos):
        if k % pasos_epoca == 0:
            r.shuffle(entrenamiento)
        filas = entrenamiento[(k % pasos_epoca) * LOTE:(k % pasos_epoca + 1) * LOTE]
        ids, mascara, accion, bio = lote(tok, filas, acciones)
        logits_accion, logits_bio = red(ids, mascara)
        perdida = perdida_ce(logits_accion, accion) + perdida_ce(logits_bio.reshape(-1, 3), bio.reshape(-1))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
        opt.step()
        plan.step()
        opt.zero_grad()
        perdidas.append(perdida.item())
        if k % 50 == 0 or k == pasos - 1:
            t = time.perf_counter() - t0
            print(f"  paso {k + 1}/{pasos} · pérdida {np.mean(perdidas[-50:]):.3f} · {t / 60:.0f} min "
                  f"(faltan ~{t / (k + 1) * (pasos - k - 1) / 60:.0f})", flush=True)
    destino = TEMPORAL / nombre
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    red.eval()
    tok.save_pretrained(destino)
    red.codificador.config.save_pretrained(destino)
    save_file({k: v.contiguous() for k, v in red.state_dict().items()}, str(destino / "interprete.safetensors"))
    (destino / "interprete.json").write_text(json.dumps({
        "base": base, "nombre": nombre, "acciones": acciones, "parametros": parametros, "temperatura": 1.0,
        "etiquetas": it.ETIQUETAS, "fecha": date.today().isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino, parametros, round((time.perf_counter() - t0) / 60, 1), round(float(np.mean(perdidas[-50:])), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidatos", default=",".join(CANDIDATOS))
    ap.add_argument("--rapido", action="store_true", help="20 pasos y validación chica (probar el script)")
    ap.add_argument("--epocas", type=float, default=EPOCAS, help="épocas (puede ser 0.5: media pasada)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    r = random.Random(20261030)
    todas = leer(RAIZ / "ml" / "intencion" / "frases_entrenamiento.csv")
    acciones = sorted({f["accion"] for f in todas})
    entrenamiento = muestra(todas, r)
    generadas = leer(RAIZ / "ml" / "intencion" / "frases_validacion_generada.csv")
    r.shuffle(generadas)
    conjuntos = {
        "validacion": (leer(RAIZ / "eval" / "frases_validacion.csv"), False),
        "objetos_validacion": (leer(RAIZ / "eval" / "frases_objetos_validacion.csv"), True),
        "comandos_validacion": (leer(RAIZ / "eval" / "frases_comandos_validacion.csv"), False),
        "generadas_objetos_reservados": (generadas[:300], True),
    }
    if args.rapido:
        conjuntos = {k: (v[:10], o) for k, (v, o) in conjuntos.items()}
    print(f"{len(entrenamiento)} frases de entrenamiento · {len(acciones)} acciones")

    tabla = []
    for nombre in args.candidatos.split(","):
        destino, parametros, minutos, perdida = entrenar(nombre, CANDIDATOS[nombre], entrenamiento, acciones, args.rapido,
                                                         args.epocas)
        interprete = it.InterpreteTransformer(destino)
        t0 = time.perf_counter()
        resultados, logits, etiquetas = evaluar(interprete, conjuntos)
        ms = (time.perf_counter() - t0) * 1000 / sum(len(f) for f, _ in conjuntos.values())
        T = ajustar_temperatura(logits, etiquetas)
        antes, despues = con_temperatura(logits, 1.0), con_temperatura(logits, T)
        acierto = antes.argmax(1) == etiquetas
        fila = {"nombre": nombre, "base": CANDIDATOS[nombre], "parametros": parametros, "minutos": minutos,
                "perdida_final": perdida, "validacion": resultados, "puntaje": round(puntaje(resultados), 4),
                "ms_por_frase": round(ms), "temperatura": round(T, 3),
                "ece_antes": round(ece(antes.max(1), acierto), 4), "ece_despues": round(ece(despues.max(1), acierto), 4)}
        tabla.append(fila)
        config = json.loads((destino / "interprete.json").read_text(encoding="utf-8"))
        (destino / "interprete.json").write_text(json.dumps({**config, "temperatura": round(T, 4),
                                                             "puntaje_validacion": fila["puntaje"]}, indent=2,
                                                            ensure_ascii=False), encoding="utf-8")
        for k, v in resultados.items():
            print(f"  {k:30s} acción {v['accion']:.3f}" + (f" · + objeto exacto {v['accion_y_objeto_exacto']:.3f}"
                                                        if v["accion_y_objeto_exacto"] is not None else ""))
        print(f"  puntaje {fila['puntaje']:.4f} · {fila['ms_por_frase']} ms por frase · temperatura {T:.2f} "
              f"(ECE {fila['ece_antes']:.3f} → {fila['ece_despues']:.3f})", flush=True)

    # El mejor en validación; a igual puntaje (0,2 puntos), el más grande. El que ya está instalado (de una
    # corrida anterior) también compite: solo se reemplaza si el nuevo le gana.
    if (SALIDA / "interprete.json").exists():
        instalado = json.loads((SALIDA / "interprete.json").read_text(encoding="utf-8"))
        if "puntaje_validacion" in instalado:
            tabla.append({"nombre": "instalado", "base": instalado["base"], "parametros": instalado["parametros"],
                          "puntaje": instalado["puntaje_validacion"]})
    mejor = max(tabla, key=lambda f: (round(f["puntaje"] / 0.002), f["parametros"]))
    if mejor["nombre"] == "instalado":
        print(f"\nse queda el instalado ({mejor['base']}): ninguno nuevo le gana en validación")
    elif not args.rapido:
        if SALIDA.exists():
            shutil.rmtree(SALIDA)
        shutil.copytree(TEMPORAL / mejor["nombre"], SALIDA)
        shutil.rmtree(TEMPORAL)
    print(f"\nelegido: {mejor['nombre']} ({mejor['base']}) -> {SALIDA.relative_to(RAIZ)}")
    if args.rapido:   # no pisa el reporte del instalado
        shutil.rmtree(TEMPORAL, ignore_errors=True)
        print("(--rapido: no se instala ni se escribe el reporte)")
        return
    if REPORTE.exists():   # se acumulan los candidatos de corridas anteriores
        anterior = json.loads(REPORTE.read_text(encoding="utf-8"))
        tabla = [f for f in anterior.get("candidatos", []) if f["nombre"] not in {g["nombre"] for g in tabla}] + tabla
    REPORTE.write_text(json.dumps({"fecha": date.today().isoformat(), "frases_entrenamiento": len(entrenamiento),
                                   "epocas": args.epocas, "lote": LOTE, "candidatos": tabla, "elegido": mejor["nombre"]},
                                  ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
