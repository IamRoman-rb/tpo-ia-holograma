"""Mide el reconocimiento de voz local (backend/voz.py, Whisper) y el sistema de punta a punta por voz.

No hay grabaciones reales del grupo todavía (ver docs/uso_responsable.md), así que se sintetiza el audio de las
frases de prueba con voces Piper que corren en el equipo (es_AR daniela, es_ES davefx, es_MX ald) y se le suma
ruido rosa a 20 dB y 10 dB de relación señal/ruido. Se mide:
  - WER (tasa de error por palabra): ediciones mínimas (Levenshtein por palabras) / palabras de la referencia,
    con el texto normalizado (minúsculas, sin tildes ni signos);
  - frase exacta: cuántas transcripciones quedan idénticas a la referencia normalizada;
  - punta a punta: la transcripción pasa por el intérprete (por defecto, el consenso) y se compara la acción con la
    de la frase escrita, con el mismo intérprete (así se aísla lo que pierde la voz);
  - palabra de activación: con "holograma, " delante, cuántas se detectan; sin ella, cuántas se activan solas.
Limitación: son voces sintéticas, más limpias que una persona en un aula; el resultado es una cota optimista.

Uso (desde la raíz, con el backend venv; tarda ~1 h con large-v3 en el procesador):
    backend/.venv/Scripts/python eval/eval_voz.py [--modelo large-v3] [--motor consenso] [--limite 40]
Salida: eval/resultados/voz_<modelo>.csv y voz_<modelo>_resumen.json. El audio queda en ml/data/externo/voz/.
"""
import argparse
import csv
import io
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))

from estadistica import formato_intervalo, intervalo_bootstrap  # noqa: E402
from texto import normalizar  # noqa: E402

AUDIO = RAIZ / "ml" / "data" / "externo" / "voz"
RESULTADOS = RAIZ / "eval" / "resultados"
VOCES = {"es_AR-daniela-high": "es/es_AR/daniela/high", "es_ES-davefx-medium": "es/es_ES/davefx/medium",
         "es_MX-ald-medium": "es/es_MX/ald/medium"}
RUIDOS = [None, 20, 10]          # dB de relación señal/ruido (None = limpio)
CONJUNTOS = ["frases_prueba.csv", "frases_nuevas.csv", "frases_comandos_prueba.csv"]


def voz_piper(nombre):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    base = VOCES[nombre]
    modelo = hf_hub_download("rhasspy/piper-voices", f"{base}/{nombre}.onnx")
    hf_hub_download("rhasspy/piper-voices", f"{base}/{nombre}.onnx.json")
    return PiperVoice.load(modelo)


def sintetizar(voz, texto, ruta):
    if ruta.exists():
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(ruta), "wb") as w:
        voz.synthesize_wav(texto, w)


def leer_wav(ruta):
    with wave.open(str(ruta), "rb") as w:
        datos = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return datos, w.getframerate()


def con_ruido(muestras, snr_db, semilla):
    """Ruido rosa (1/f, más parecido al de un ambiente que el blanco) con la relación señal/ruido pedida."""
    if snr_db is None:
        return muestras
    gen = np.random.default_rng(semilla)
    blanco = np.fft.rfft(gen.normal(size=len(muestras)))
    f = np.arange(len(blanco))
    rosa = np.fft.irfft(blanco / np.sqrt(np.maximum(f, 1)), n=len(muestras)).astype(np.float32)
    potencia = np.mean(muestras ** 2)
    rosa *= np.sqrt(potencia / (np.mean(rosa ** 2) * 10 ** (snr_db / 10)))
    return np.clip(muestras + rosa, -1, 1)


def a_wav(muestras, frecuencia):
    salida = io.BytesIO()
    with wave.open(salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(frecuencia)
        w.writeframes((muestras * 32767).astype("<i2").tobytes())
    return salida.getvalue()


def wer(referencia, hipotesis):
    r, h = normalizar(referencia).replace(",", " ").replace(".", " ").split(), \
        normalizar(hipotesis).replace(",", " ").replace(".", " ").split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        anterior, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            anterior, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, anterior + (r[i - 1] != h[j - 1]))
    return d[len(h)], max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="large-v3")
    ap.add_argument("--motor", default="consenso", help="intérprete para la medida de punta a punta")
    ap.add_argument("--limite", type=int, default=0, help="frases por conjunto (0 = todas)")
    args = ap.parse_args()

    import intent
    import voz
    whisper = voz.cargar(args.modelo)
    frases = []
    for nombre in CONJUNTOS:
        with open(RAIZ / "eval" / nombre, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        frases += [(nombre, fila["frase"], fila["accion"]) for fila in (filas[:args.limite] if args.limite else filas)]
    print(f"{len(frases)} frases × {len(VOCES)} voces × {len(RUIDOS)} niveles de ruido · Whisper {args.modelo}")

    # La interpretación de la frase escrita, una vez por frase (referencia de punta a punta)
    escrita = {f: intent.interpretar(f, args.motor)["accion"] for _, f, _ in frases}

    filas, t0 = [], time.perf_counter()
    for nombre_voz in VOCES:
        piper = voz_piper(nombre_voz)
        for k, (conjunto, frase, accion) in enumerate(frases):
            for activacion in (False, True):
                if activacion and (k % 3 or nombre_voz != "es_AR-daniela-high"):
                    continue   # la palabra de activación se prueba con una de cada tres frases y una voz
                texto = f"Holograma, {frase}" if activacion else frase
                ruta = AUDIO / nombre_voz / f"{conjunto[7:-4]}_{k:03d}{'_activacion' if activacion else ''}.wav"
                sintetizar(piper, texto, ruta)
                muestras, fr = leer_wav(ruta)
                for snr in (RUIDOS if not activacion else [None]):
                    r = voz.transcribir(a_wav(con_ruido(muestras, snr, k), fr), modelo=whisper)
                    activado, resto = voz.quitar_activacion(r["texto"])
                    errores, palabras = wer(frase, resto if activacion else r["texto"])
                    hipotesis = resto if activado else r["texto"]
                    accion_voz = intent.interpretar(hipotesis, args.motor)["accion"] if hipotesis else "desconocido"
                    filas.append({"voz": nombre_voz, "conjunto": conjunto, "frase": frase, "ruido_db": snr or "",
                                  "con_activacion": activacion, "transcripcion": r["texto"], "errores": errores,
                                  "palabras": palabras, "exacta": errores == 0, "activado": activado,
                                  "accion_escrita": escrita[frase], "accion_voz": accion_voz,
                                  "accion_esperada": accion, "ms": r["ms_transcripcion"]})
            if k % 10 == 0:
                print(f"  {nombre_voz} {k + 1}/{len(frases)} · {(time.perf_counter() - t0) / 60:.0f} min", flush=True)

    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / f"voz_{args.modelo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    resumen = {"modelo": args.modelo, "motor": args.motor, "frases": len(frases), "por_condicion": []}
    print()
    sin_act = [f for f in filas if not f["con_activacion"]]
    for nombre_voz in VOCES:
        for snr in RUIDOS:
            g = [f for f in sin_act if f["voz"] == nombre_voz and (f["ruido_db"] or None) == snr]
            if not g:
                continue
            w_ = sum(f["errores"] for f in g) / sum(f["palabras"] for f in g)
            igual = [f["accion_voz"] == f["accion_escrita"] for f in g]
            correcta = [f["accion_voz"] == f["accion_esperada"] for f in g]
            fila = {"voz": nombre_voz, "ruido_db": snr, "wer": round(w_, 4),
                    "frase_exacta": round(np.mean([f["exacta"] for f in g]), 4),
                    "accion_igual_que_escrita": round(float(np.mean(igual)), 4),
                    "accion_correcta": [round(v, 4) for v in intervalo_bootstrap(correcta)],
                    "ms_mediana": statistics.median(f["ms"] for f in g)}
            resumen["por_condicion"].append(fila)
            print(f"  {nombre_voz:22s} {('limpio' if snr is None else f'{snr} dB'):>7s} · WER {100 * w_:5.1f} % · "
                  f"exacta {100 * fila['frase_exacta']:3.0f} % · acción como escrita {100 * fila['accion_igual_que_escrita']:3.0f} % · "
                  f"acción correcta {formato_intervalo(*intervalo_bootstrap(correcta))} · {fila['ms_mediana']:.0f} ms")
    act = [f for f in filas if f["con_activacion"]]
    falsas = [f["activado"] for f in sin_act]
    resumen["activacion"] = {"detectadas": round(float(np.mean([f["activado"] for f in act])), 4) if act else None,
                             "falsas_activaciones": round(float(np.mean(falsas)), 4)}
    print(f"\n  palabra de activación: detectada {100 * resumen['activacion']['detectadas']:.0f} % · "
          f"falsas activaciones {100 * resumen['activacion']['falsas_activaciones']:.1f} %")
    (RESULTADOS / f"voz_{args.modelo}_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
