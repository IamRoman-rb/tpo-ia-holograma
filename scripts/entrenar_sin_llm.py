"""Entrena todos los modelos menos el LLM propio, sin tocar el entrenamiento del LLM que esté en pausa.

Uso (desde la raíz del repositorio, con cualquier Python 3.10+ de 64 bits):
    python scripts/entrenar_sin_llm.py --estado               # cómo va el LLM pausado y cada uno de los demás modelos
    python scripts/entrenar_sin_llm.py --listar               # qué pasos corre y cuánto tarda cada uno
    python scripts/entrenar_sin_llm.py                        # todo menos el LLM (~18 h en CPU, 16 son del transformer)
    python scripts/entrenar_sin_llm.py --saltear transformer  # todo menos el LLM y el transformer (~1,5 h)
    python scripts/entrenar_sin_llm.py --solo red,arboles     # solo esos pasos
    python scripts/entrenar_sin_llm.py --prueba               # versión mínima de cada paso
    python scripts/entrenar_sin_llm.py --con-neuroevolucion   # suma el algoritmo genético de gestos (~2,5 h)

Pausa: igual que scripts/entrenar.py (tecla P, --horas N, o `python scripts/entrenar.py --pausar` desde otra
terminal), pero se retoma con  python scripts/entrenar_sin_llm.py --continuar.
Su estado va en ml/checkpoints/entrenamiento_sin_llm.json, aparte de ml/checkpoints/entrenamiento.json (el de
scripts/entrenar.py). Así el LLM pausado se sigue retomando con  python scripts/entrenar.py --continuar.

Lo que este script no hace a propósito:
  - No corre los pasos «voz» ni «frases». Reescriben ml/intencion/frases_entrenamiento.csv, y si cambia ese
    archivo, el punto de control del LLM ya no sirve y el LLM arranca de cero.
  - No borra ni reescribe ml/checkpoints/entrenamiento.json ni ml/checkpoints/llm_*.pt.
  - No corre a la vez que el LLM salvo con --en-paralelo. Los dos comparten la CPU y la RAM (el 7B necesita mucha),
    y comparten la señal de pausa: la tecla P en cualquiera de los dos frena a ambos.

«consenso» y «evaluar» usan el LLM propio que esté instalado en backend/modelos/ (hoy el que haya terminado; el 7B
recién cuando termine). Después de que termine el LLM conviene repetirlos; scripts/entrenar.py --continuar ya los
tiene en la lista de pendientes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar  # noqa: E402  (trae levantar y ml/pausa.py; no importa torch)
from entrenar import LOGS, RAIZ, duracion, pausa  # noqa: E402

ESTADO = pausa.CARPETA / "entrenamiento_sin_llm.json"
ESTADO_LLM = entrenar.ESTADO
RETOMAR = "python scripts/entrenar_sin_llm.py --continuar"
PROHIBIDOS = {"llm": "es el LLM: se entrena con scripts/entrenar.py",
              "voz": "reescribe las frases y el LLM pausado arrancaría de cero",
              "frases": "reescribe las frases y el LLM pausado arrancaría de cero"}


def todos_los_pasos(llm):
    return [p for p in entrenar.pasos(llm) if p.nombre not in PROHIBIDOS]


# ------------------------------------------------------------------ procesos que ya están entrenando
def entrenamientos_corriendo():
    """Líneas de comando de los entrenamientos de Python que están corriendo (menos este proceso)."""
    try:
        if os.name == "nt":
            consulta = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
            salida = subprocess.run(["powershell", "-NoProfile", "-Command", consulta], capture_output=True,
                                    text=True, timeout=30).stdout
        else:
            salida = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    propios = {os.getpid(), os.getppid()}
    encontrados = []
    for linea in salida.splitlines():
        pid, _, comando = linea.strip().replace("\t", " ").partition(" ")
        if not pid.isdigit() or int(pid) in propios:
            continue
        if re.search(r"entrenar(_\w+)?\.py|ajustar_consenso|eval_intent|neuroevolucion", comando):
            encontrados.append(comando.strip())
    return encontrados


def llm_corriendo(comandos):
    return any("entrenar_llm.py" in c or re.search(r"scripts[\\/]entrenar\.py", c) for c in comandos)


# ------------------------------------------------------------------ --estado
def ultima_modificacion(salida):
    ruta = RAIZ / salida.split(" ")[0]
    if not ruta.exists():
        return None
    if ruta.is_dir():
        fechas = [f.stat().st_mtime for f in ruta.rglob("*") if f.is_file()]
        return max(fechas) if fechas else None
    return ruta.stat().st_mtime


def avance_llm(llm):
    """Último paso, total, horas entrenadas y minutos que faltan según los logs del LLM (sin los de prueba)."""
    logs = sorted((f for f in LOGS.glob("*_llm.log") if f.stat().st_size > 0), key=lambda f: f.name)
    paso = total = horas = faltan = None
    for log in logs:
        texto = log.read_text(encoding="utf-8", errors="replace")
        if f"--tamano {llm}" not in texto.splitlines()[0]:
            continue
        for m in re.finditer(r"paso (\d+)/(\d+)", texto):
            paso, total = int(m[1]), int(m[2])
        for m in re.finditer(r"guardado: paso (\d+)", texto):
            paso = max(paso or 0, int(m[1]))
        for m in re.finditer(r"\(([\d.]+) h ya entrenadas\)", texto):
            horas = float(m[1])
        for m in re.finditer(r"paso (\d+)/(\d+) .*?faltan ~(\d+)", texto):
            # lo que faltaba en ese paso, llevado al último paso guardado
            faltan = int(m[3]) * (int(m[2]) - (paso or int(m[1]))) / max(int(m[2]) - int(m[1]), 1)
    return paso, total, horas, faltan


def mostrar_estado():
    print("LLM propio (scripts/entrenar.py)")
    if ESTADO_LLM.exists():
        estado = json.loads(ESTADO_LLM.read_text(encoding="utf-8"))
        llm = estado["llm"]
        print(f"  en pausa desde el {estado['fecha'].replace('T', ' a las ')} · Qwen {llm}"
              f"{' (prueba)' if estado['prueba'] else ''}")
        print(f"  pasos pendientes de esa corrida: {', '.join(estado['pendientes'])}")
        punto = pausa.CARPETA / f"llm_{llm}{'_prueba' if estado['prueba'] else ''}.pt"
        if punto.exists():
            print(f"  punto de control: {punto.relative_to(RAIZ).as_posix()} ({punto.stat().st_size / 2**30:.1f} GB, "
                  f"{datetime.fromtimestamp(punto.stat().st_mtime):%Y-%m-%d %H:%M})")
        paso, total, horas, faltan = avance_llm(llm)
        if paso and total:
            texto = f"  avance: paso {paso}/{total} ({paso / total:.0%})"
            if horas:
                texto += f" · más de {horas:.0f} h entrenadas"
            if faltan:
                texto += f" · faltan {duracion(round(faltan))}"
            print(texto)
        print("  para seguirlo: python scripts/entrenar.py --continuar")
    else:
        print("  no hay ningún entrenamiento del LLM en pausa")

    corriendo = entrenamientos_corriendo()
    print("\nCorriendo ahora")
    print("  " + ("\n  ".join(corriendo) if corriendo else "nada"))

    print("\nTodo menos el LLM (scripts/entrenar_sin_llm.py)")
    if ESTADO.exists():
        estado = json.loads(ESTADO.read_text(encoding="utf-8"))
        print(f"  en pausa desde el {estado['fecha'].replace('T', ' a las ')}: "
              f"falta {', '.join(estado['pendientes'])}. Para seguir: {RETOMAR}")
    else:
        print("  no hay nada en pausa")
    print(f"\n  {'paso':15s} {'última salida':17s}  archivo")
    for p in todos_los_pasos("maximo"):
        fecha = ultima_modificacion(p.salida[0]) if p.salida else None
        cuando = f"{datetime.fromtimestamp(fecha):%Y-%m-%d %H:%M}" if fecha else "nunca"
        print(f"  {p.nombre:15s} {cuando:17s}  {p.salida[0] if p.salida else ''}")


# ------------------------------------------------------------------ entrenamiento
def guardar_estado(pendientes, args):
    pausa.escribir_json(ESTADO, {"pendientes": [p.nombre for p in pendientes], "prueba": args.prueba,
                                 "fecha": datetime.now().isoformat(timespec="seconds")})


def elegir(args, llm):
    todos = todos_los_pasos(llm)
    nombres = [p.nombre for p in todos]
    pedidos = {s.strip() for s in (args.solo or "").split(",") if s.strip()}
    salteados = {s.strip() for s in (args.saltear or "").split(",") if s.strip()}
    for valor in pedidos | salteados:
        if valor in PROHIBIDOS:
            sys.exit(f"«{valor}» no se corre desde acá: {PROHIBIDOS[valor]}.")
        if valor not in nombres:
            sys.exit(f"No existe el paso «{valor}». Pasos: {', '.join(nombres)}")
    if pedidos:
        elegidos = [p for p in todos if p.nombre in pedidos]
    else:
        elegidos = [p for p in todos if p.por_defecto or (args.con_neuroevolucion and p.nombre == "neuroevolucion")]
    elegidos = [p for p in elegidos if p.nombre not in salteados]
    if args.prueba:
        elegidos = [p for p in elegidos if p.minutos_prueba]
    return elegidos


def main():
    ap = argparse.ArgumentParser(description="Entrena todo menos el LLM propio, sin tocar el LLM pausado.")
    ap.add_argument("--estado", action="store_true", help="mostrar cómo va cada entrenamiento y salir")
    ap.add_argument("--listar", action="store_true", help="mostrar los pasos y salir")
    ap.add_argument("--solo", help="pasos separados por coma (ej. red,arboles)")
    ap.add_argument("--saltear", help="pasos a no correr, separados por coma (ej. transformer)")
    ap.add_argument("--con-neuroevolucion", action="store_true", help="sumar el algoritmo genético de gestos")
    ap.add_argument("--prueba", action="store_true", help="versión mínima de cada paso")
    ap.add_argument("--continuar", action="store_true", help="retomar lo que quedó en pausa")
    ap.add_argument("--desde-cero", action="store_true", help="descartar lo que quedó en pausa (no toca el LLM)")
    ap.add_argument("--horas", type=float, help="pausar solo después de tantas horas")
    ap.add_argument("--en-paralelo", action="store_true",
                    help="correr aunque el LLM esté entrenando (comparten CPU, RAM y la tecla de pausa)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # los sys.exit con mensaje van por stderr

    if args.estado:
        mostrar_estado()
        return
    if sys.version_info < (3, 10) or struct.calcsize("P") != 8:
        sys.exit("Hace falta Python 3.10 o más nuevo, de 64 bits (PyTorch no tiene versión de 32 bits).")

    llm = json.loads(ESTADO_LLM.read_text(encoding="utf-8"))["llm"] if ESTADO_LLM.exists() else "maximo"
    if args.listar:
        print(f"{'paso':15s} {'tarda':>8s} {'prueba':>8s}  qué entrena")
        for p in todos_los_pasos(llm):
            motivo = entrenar.motivo_para_saltear(p)
            extra = f"  [no se puede: {motivo}]" if motivo else ("" if p.por_defecto else "  [con --con-neuroevolucion]")
            prueba = duracion(p.minutos_prueba) if p.minutos_prueba else "-"
            print(f"{p.nombre:15s} {duracion(p.minutos):>8s} {prueba:>8s}  {p.que}{extra}")
        return

    if args.continuar:
        if args.desde_cero or args.solo or args.saltear or args.prueba or args.con_neuroevolucion:
            sys.exit("--continuar usa las opciones de la corrida pausada: solo se combina con --horas y --en-paralelo.")
        if not ESTADO.exists():
            sys.exit("No hay nada en pausa para continuar.")
        estado = json.loads(ESTADO.read_text(encoding="utf-8"))
        args.prueba = estado["prueba"]
        a_elegir = [p for p in todos_los_pasos(llm) if p.nombre in estado["pendientes"]]
        print(f"Continuando lo pausado el {estado['fecha'].replace('T', ' a las ')}: "
              f"falta {', '.join(p.nombre for p in a_elegir)}.")
    else:
        if ESTADO.exists() and not args.desde_cero:
            estado = json.loads(ESTADO.read_text(encoding="utf-8"))
            sys.exit(f"Hay una corrida en pausa desde el {estado['fecha'].replace('T', ' a las ')} "
                     f"(falta {', '.join(estado['pendientes'])}).\n  Para seguirla:   {RETOMAR}\n"
                     "  Para descartarla: python scripts/entrenar_sin_llm.py --desde-cero [otras opciones]")
        if args.desde_cero:
            ESTADO.unlink(missing_ok=True)
            for nombre in ("transformer.pt", "transformer_prueba.pt"):   # solo el avance del transformer
                (pausa.CARPETA / nombre).unlink(missing_ok=True)
            print("Se descartó lo pausado de este script (el LLM pausado sigue intacto).")
        a_elegir = elegir(args, llm)

    corriendo = entrenamientos_corriendo()
    if corriendo:
        if not llm_corriendo(corriendo) or not args.en_paralelo:
            sys.exit("Ya hay un entrenamiento corriendo:\n  " + "\n  ".join(corriendo) +
                     ("\nSi es el LLM y querés entrenar a la vez igual, agregá --en-paralelo "
                      "(comparten CPU y RAM, y la tecla P frena a los dos)." if llm_corriendo(corriendo) else ""))
        print("Atención: el LLM está entrenando a la vez. Cada uno va a ir más lento, y pausar uno pausa al otro.")
    en_paralelo = bool(corriendo)

    saltear = {p.nombre: m for p in a_elegir if (m := entrenar.motivo_para_saltear(p))}
    a_correr = [p for p in a_elegir if p.nombre not in saltear]
    if not a_correr:
        sys.exit("No queda ningún paso para correr" +
                 (": " + "; ".join(f"{n}: {m}" for n, m in saltear.items()) if saltear else "."))

    entrenar.preparar_entorno()
    entrenar.instalar(entrenar.EXTRA, entrenar.HUELLA_EXTRA, "las dependencias de entrenamiento")
    LOGS.mkdir(parents=True, exist_ok=True)
    entrenar.no_suspender()

    total = sum(p.minutos_prueba if args.prueba else p.minutos for p in a_correr)
    print(f"\n{'Prueba' if args.prueba else 'Entrenamiento'} sin el LLM: {len(a_correr)} pasos "
          f"({', '.join(p.nombre for p in a_correr)}), {duracion(total)} en CPU.")
    for nombre, motivo in saltear.items():
        print(f"  se saltea {nombre}: {motivo}")
    if any(p.nombre in ("consenso", "evaluar") for p in a_correr):
        print("  consenso y evaluar usan el LLM propio que ya esté instalado; repetilos cuando termine el LLM.")

    if not en_paralelo:
        pausa.cancelar()   # un pedido viejo no tiene que frenar esta corrida (si el LLM corre, puede ser suyo)
    teclado = "Apretá P para pausar" if sys.stdin.isatty() else "Para pausar: python scripts/entrenar.py --pausar"
    print(f"{teclado}; después, {RETOMAR}" + (f". Se pausa solo a las {args.horas:g} h." if args.horas else "."))
    threading.Thread(target=entrenar.escuchar_pausa, args=(args.horas,), daemon=True).start()

    resumen, pausado = [], None
    for i, paso in enumerate(a_correr, 1):
        if pausa.pedida():
            pausado = a_correr[i - 1:]
            break
        entrenar.paso_actual = paso
        print(f"\n[{i}/{len(a_correr)}] {paso.nombre}: {paso.que} "
              f"({duracion(paso.minutos_prueba if args.prueba else paso.minutos)})", flush=True)
        log = LOGS / f"{datetime.now():%Y-%m-%d_%H%M}_{paso.nombre}{'_prueba' if args.prueba else ''}.log"
        t0 = time.perf_counter()
        try:
            codigo = entrenar.correr(paso, args.prueba, log)
        except KeyboardInterrupt:
            guardar_estado(a_correr[i - 1:], args)
            sys.exit(f"\nCortado en «{paso.nombre}». Para retomar: {RETOMAR}")
        minutos = (time.perf_counter() - t0) / 60
        if codigo == pausa.CODIGO:
            resumen.append((paso.nombre, "pausa", minutos))
            pausado = a_correr[i - 1:]
            break
        resumen.append((paso.nombre, "ok   " if codigo == 0 else "FALLÓ", minutos))
        if codigo != 0:
            guardar_estado(a_correr[i - 1:], args)
            print(f"\nFalló «{paso.nombre}» (código {codigo}). La salida completa está en "
                  f"{log.relative_to(RAIZ).as_posix()}.\nCuando lo arregles: {RETOMAR}")
            break
    entrenar.paso_actual = None

    print("\nResumen:")
    for nombre, como, minutos in resumen:
        print(f"  {como} {nombre:15s} {minutos:6.1f} min")
    if pausado:
        guardar_estado(pausado, args)
        if not en_paralelo:
            pausa.cancelar()
        print(f"\nEn pausa. Falta: {', '.join(p.nombre for p in pausado)}.\nPara seguir: {RETOMAR}")
        return
    if all(c == "ok   " for _, c, _ in resumen) and len(resumen) == len(a_correr):
        ESTADO.unlink(missing_ok=True)
        print("\nListo: todo menos el LLM quedó entrenado.")
        if ESTADO_LLM.exists():
            print("El LLM sigue en pausa: python scripts/entrenar.py --continuar")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
