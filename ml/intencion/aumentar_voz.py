"""Frases de entrenamiento con los errores reales de la voz: se dicen con voces sintéticas, con ruido, y se transcriben
con Whisper, igual que en eval/eval_voz.py (y con sus mismas funciones). La transcripción conserva la acción.

Por voz se acierta ~10 puntos menos que escribiendo (docs/tecnicas_ia.md): Whisper cambia palabras ("rulemán" ->
"ruleman", "explotá" -> "explota") y los intérpretes nunca vieron esos errores. Acá los ven.

  - Solo frases de ENTRENAMIENTO (ml/intencion/frases_entrenamiento.csv, fuente "plantilla"), elegidas al azar con
    la misma cantidad por acción: ninguna de evaluación.
  - Una voz Piper al azar (es_AR, es_ES, es_MX) y un nivel de ruido al azar (limpio, 20 dB, 10 dB).
  - El objeto: el tramo de la transcripción más parecido al objeto original (distancia de edición por letras). Si
    Whisper lo cambió tanto que no se reconoce (más de 40 % de la palabra), la frase se descarta: el etiquetador
    aprendería a marcar algo que no está.
Se escribe a medida que avanza: se puede cortar y volver a correr (sigue donde quedó).

Uso (desde la raíz; ~8 s por frase con large-v3):
    backend/.venv/Scripts/python ml/intencion/aumentar_voz.py [--frases 1500] [--modelo large-v3]
Después: generar_frases.py las suma al entrenamiento (fuente "voz").
"""
import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))
from eval_voz import RUIDOS, VOCES, a_wav, con_ruido, leer_wav, sintetizar, voz_piper  # noqa: E402
from texto import normalizar  # noqa: E402

SALIDA = AQUI / "frases_voz.csv"
AUDIO = RAIZ / "ml" / "data" / "externo" / "voz_entrenamiento"
CAMPOS = ["frase", "accion", "objeto", "frase_original", "objeto_original", "voz", "ruido_db", "clave"]
MAX_DISTANCIA = 0.4


def distancia(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        anterior, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            anterior, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, anterior + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def alinear_objeto(transcripcion, objeto):
    """El tramo de palabras de la transcripción más parecido al objeto, o None si ninguno se parece lo suficiente."""
    palabras = transcripcion.split()
    meta = normalizar(objeto)
    n = len(meta.split())
    mejor, d_mejor = None, None
    for largo in {max(1, n - 1), n, n + 1}:
        for i in range(len(palabras) - largo + 1):
            tramo = " ".join(palabras[i:i + largo]).strip(" ,.?¿!¡")
            d = distancia(normalizar(tramo), meta)
            if d_mejor is None or d < d_mejor:
                mejor, d_mejor = tramo, d
    return mejor if mejor and d_mejor <= MAX_DISTANCIA * len(meta) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frases", type=int, default=1500)
    ap.add_argument("--modelo", default="large-v3")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = random.Random(20261030)

    with open(AQUI / "frases_entrenamiento.csv", encoding="utf-8") as f:
        plantillas = [x for x in csv.DictReader(f) if x.get("fuente", "plantilla") == "plantilla"]
    por_accion = defaultdict(list)
    for x in plantillas:
        por_accion[x["accion"]].append(x)
    cupo = args.frases // len(por_accion)
    elegidas = [x for lista in por_accion.values() for x in r.sample(lista, min(cupo, len(lista)))]
    r.shuffle(elegidas)
    tareas = [(f"{k:05d}", x, r.choice(list(VOCES)), r.choice(RUIDOS)) for k, x in enumerate(elegidas)]

    hechas = set()
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8") as f:
            hechas = {x["clave"] for x in csv.DictReader(f)}
    else:
        with open(SALIDA, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CAMPOS).writeheader()
    pendientes = [t for t in tareas if t[0] not in hechas]
    print(f"{len(tareas)} frases ({cupo} por acción) · faltan {len(pendientes)} · Whisper {args.modelo}", flush=True)

    import voz
    whisper = voz.cargar(args.modelo)
    voces, t0, descartadas = {}, time.perf_counter(), 0
    for i, (clave, x, nombre_voz, snr) in enumerate(pendientes, 1):
        if nombre_voz not in voces:
            voces[nombre_voz] = voz_piper(nombre_voz)
        ruta = AUDIO / nombre_voz / f"{clave}.wav"
        sintetizar(voces[nombre_voz], x["frase"], ruta)
        muestras, frecuencia = leer_wav(ruta)
        texto = voz.transcribir(a_wav(con_ruido(muestras, snr, int(clave)), frecuencia), modelo=whisper)["texto"].strip()
        objeto = alinear_objeto(texto, x["objeto"]) if x["objeto"] else ""
        if texto and objeto is not None:
            with open(SALIDA, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CAMPOS).writerow({
                    "frase": texto, "accion": x["accion"], "objeto": objeto, "frase_original": x["frase"],
                    "objeto_original": x["objeto"], "voz": nombre_voz, "ruido_db": snr or "", "clave": clave})
        else:
            descartadas += 1
        if i % 25 == 0 or i == len(pendientes):
            minutos = (time.perf_counter() - t0) / 60
            print(f"  {i}/{len(pendientes)} · {descartadas} descartadas · {minutos:.0f} min "
                  f"(faltan ~{minutos / i * (len(pendientes) - i):.0f})", flush=True)


if __name__ == "__main__":
    main()
