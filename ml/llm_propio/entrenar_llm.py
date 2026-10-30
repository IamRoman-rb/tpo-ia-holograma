"""Ajusta un LLM abierto chico para interpretar pedidos: el LLM propio del proyecto (backend/llm_propio.py).

Base: Qwen2.5-Instruct (Alibaba, licencia Apache 2.0), que ya sabe español. Dos tamaños: el de 1.500 millones
de parámetros ("grande", el preferido: más preciso) y el de 500 millones ("chico", tres veces más rápido).
Se ajusta con LoRA: se congelan los pesos originales y se entrenan matrices chicas que se suman a cada capa
(8,8 millones de parámetros, 1,7 % del modelo). Al terminar se suman a los pesos y queda un modelo común.

Aprende a escribir, para cada pedido, "accion|objeto" (formato corto: en el procesador cada token cuesta
~70 ms; el backend lo convierte al JSON de siempre). La vista y el factor de escala los siguen sacando las
reglas, como en el intérprete entrenado (backend/intent_modelo.py).

Datos: ml/intencion/frases_entrenamiento.csv (las mismas frases del intérprete entrenado), una muestra
balanceada por acción: en el procesador, una pasada por las 18.800 frases lleva casi 3 horas.
Se elige con validación (frases_validacion.csv, frases_objetos_validacion.csv y 200 frases generadas con
objetos reservados) y se mide de punta a punta con eval/eval_intent.py --motor llm_propio.

Uso (desde la raíz):
    backend/.venv/Scripts/python ml/llm_propio/entrenar_llm.py --tamano grande   # ~4 h en CPU
    backend/.venv/Scripts/python ml/llm_propio/entrenar_llm.py --tamano chico    # ~80 min
Requiere: pip install peft (solo para entrenar; el backend usa el modelo ya combinado)
Salida: backend/modelos/llm_propio_grande/ (~3 GB) o llm_propio/ (~1 GB), que no se versionan, y
ml/resultados/llm_propio_<tamaño>_reporte.json
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

import torch

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
import llm_propio  # noqa: E402  (formato de entrada y salida: el mismo que usa el backend)

BASES = {"chico": "Qwen/Qwen2.5-0.5B-Instruct", "grande": "Qwen/Qwen2.5-1.5B-Instruct"}
POR_ACCION = {"mostrar": 4000, "explotar": 2400, "aislar": 2400}   # el resto: 900 cada una (el doble desde
OTRAS = 900                                                       # que hay 100 mil frases, con tipeo y voz)
LOTE = 16
TASA = 2e-4


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


def lote_de_entrenamiento(tok, filas):
    """Entrada + respuesta; la pérdida se calcula solo sobre la respuesta (la entrada se marca con -100)."""
    ids, etiquetas = [], []
    for f in filas:
        entrada = tok(llm_propio.entrada(f["frase"])).input_ids
        respuesta = tok(llm_propio.respuesta(f["accion"], f["objeto"])).input_ids + [tok.eos_token_id]
        ids.append(entrada + respuesta)
        etiquetas.append([-100] * len(entrada) + respuesta)
    largo = max(map(len, ids))
    relleno = tok.pad_token_id
    return (torch.tensor([x + [relleno] * (largo - len(x)) for x in ids]),
            torch.tensor([[1] * len(x) + [0] * (largo - len(x)) for x in ids]),
            torch.tensor([y + [-100] * (largo - len(y)) for y in etiquetas]))


def evaluar(interprete, filas, con_objeto):
    """Exactitud de la acción y, si el set tiene el objeto exacto, de acción + objeto."""
    acc = completo = 0
    for f in filas:
        r = interprete.generar(f["frase"])
        acc += r["accion"] == f["accion"]
        if con_objeto:
            completo += (r["accion"] == f["accion"]
                         and llm_propio.palabras(r["objeto"] or "") == llm_propio.palabras(f["objeto"]))
    n = len(filas)
    return round(acc / n, 4), (round(completo / n, 4) if con_objeto else None)


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="40 pasos y validación chica, para probar el script")
    ap.add_argument("--tamano", choices=list(BASES), default="grande")
    args = ap.parse_args()
    BASE, SALIDA = BASES[args.tamano], llm_propio.RUTAS[args.tamano]
    REPORTE = RAIZ / "ml" / "resultados" / f"llm_propio_{args.tamano}_reporte.json"
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads() * 2))
    r = random.Random(20261029)

    entrenamiento = muestra(leer(RAIZ / "ml" / "intencion" / "frases_entrenamiento.csv"), r)
    generadas = leer(RAIZ / "ml" / "intencion" / "frases_validacion_generada.csv")
    r.shuffle(generadas)
    validacion = {
        "validacion": (leer(RAIZ / "eval" / "frases_validacion.csv"), False),
        "objetos_validacion": (leer(RAIZ / "eval" / "frases_objetos_validacion.csv"), True),
        "comandos_validacion": (leer(RAIZ / "eval" / "frases_comandos_validacion.csv"), False),
        "generadas_objetos_reservados": (generadas[:200], True),
    }
    if args.rapido:   # no pisa el modelo instalado ni su reporte
        entrenamiento = entrenamiento[:40 * LOTE]
        validacion = {k: (v[:15], o) for k, (v, o) in validacion.items()}
        SALIDA = SALIDA.with_name(SALIDA.name + "_prueba")
    pasos = math.ceil(len(entrenamiento) / LOTE)
    print(f"{len(entrenamiento)} frases de entrenamiento · {pasos} pasos de {LOTE}")

    tok = AutoTokenizer.from_pretrained(BASE)
    modelo = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"parámetros entrenables: {entrenables:,} de {total:,} ({100 * entrenables / total:.1f} %)")

    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=TASA, weight_decay=0.0)
    calentamiento = max(1, pasos // 20)
    plan = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1, (k + 1) / calentamiento)
                                             * 0.5 * (1 + math.cos(math.pi * min(k, pasos) / pasos)))
    modelo.train()
    t0, perdidas = time.perf_counter(), []
    for k in range(pasos):
        ids, mascara, etiquetas = lote_de_entrenamiento(tok, entrenamiento[k * LOTE:(k + 1) * LOTE])
        perdida = modelo(input_ids=ids, attention_mask=mascara, labels=etiquetas).loss
        perdida.backward()
        torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
        opt.step()
        plan.step()
        opt.zero_grad()
        perdidas.append(perdida.item())
        if k % 25 == 0 or k == pasos - 1:
            transcurrido = time.perf_counter() - t0
            print(f"  paso {k + 1}/{pasos} · pérdida {sum(perdidas[-25:]) / len(perdidas[-25:]):.3f} · "
                  f"{transcurrido / 60:.0f} min (faltan ~{transcurrido / (k + 1) * (pasos - k - 1) / 60:.0f})",
                  flush=True)
    minutos = round((time.perf_counter() - t0) / 60, 1)

    # Combinar LoRA con los pesos originales y guardar en float16 (la mitad de tamaño; se carga en float32)
    modelo = modelo.merge_and_unload()
    modelo.eval()
    SALIDA.mkdir(parents=True, exist_ok=True)
    modelo.to(torch.float16).save_pretrained(SALIDA)
    tok.save_pretrained(SALIDA)
    tamano = sum(f.stat().st_size for f in SALIDA.iterdir()) // 2**20
    print(f"modelo -> {SALIDA.relative_to(RAIZ)} ({tamano} MB)")

    interprete = llm_propio.LlmPropio(SALIDA)
    resultados, t1 = {}, time.perf_counter()
    for nombre, (filas, con_objeto) in validacion.items():
        accion, completo = evaluar(interprete, filas, con_objeto)
        resultados[nombre] = {"frases": len(filas), "accion": accion, "accion_y_objeto_exacto": completo}
        print(f"  {nombre:30s} acción {accion:.3f}" + (f" · acción + objeto exacto {completo:.3f}" if con_objeto else ""))
    n = sum(len(f) for f, _ in validacion.values())
    ms = round((time.perf_counter() - t1) * 1000 / n)
    print(f"  {ms} ms por frase")
    if args.rapido:
        del interprete
        shutil.rmtree(SALIDA, ignore_errors=True)
        print("(--rapido: se borra el modelo de prueba y no se escribe el reporte)")
        return

    REPORTE.parent.mkdir(exist_ok=True)
    REPORTE.write_text(json.dumps({
        "fecha": date.today().isoformat(), "tamano": args.tamano, "base": BASE, "lora": {"r": 16, "alpha": 32, "capas": "todas las lineales"},
        "parametros_entrenables": entrenables, "parametros_totales": total,
        "frases_entrenamiento": len(entrenamiento), "pasos": pasos, "lote": LOTE, "tasa": TASA,
        "minutos_entrenamiento": minutos, "perdida_final": round(sum(perdidas[-25:]) / len(perdidas[-25:]), 4),
        "validacion": resultados, "ms_por_frase": ms, "tamano_mb": tamano,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
