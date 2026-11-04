"""Entrena los modelos del proyecto con los datos del repositorio, en orden y en una sola terminal.

Uso (desde la raíz del repositorio, con cualquier Python 3.10+ de 64 bits):
    python scripts/entrenar.py --listar            # qué pasos hay, cuánto tarda cada uno y cuáles se pueden correr
    python scripts/entrenar.py --prueba            # ~20 min: cada paso en su versión mínima, sin pisar los modelos lentos
    python scripts/entrenar.py                     # configuraci?n m?xima; puede superar 2 d?as en CPU
    python scripts/entrenar.py --llm grande        # Qwen 1.5B si no alcanza la memoria para 7B
    python scripts/entrenar.py --solo bayes,modelo # solo esos pasos
    python scripts/entrenar.py --desde llm         # empezar en ese paso
    python scripts/entrenar.py --con-voz           # antes, rehacer las frases con voz de Whisper (~3,5 h más)

Pausa (para entrenar en tandas en vez de 3 días seguidos):
    Tecla P mientras entrena                      # termina el paso de optimización en curso, guarda y frena
                                                  # (P otra vez antes de que frene: se cancela la pausa)
    python scripts/entrenar.py --horas 8          # lo mismo, solo, después de 8 horas
    python scripts/entrenar.py --pausar           # desde otra terminal, si esta no recibe teclas
    python scripts/entrenar.py --continuar        # retoma donde quedó, con las mismas opciones
    python scripts/entrenar.py --continuar --llm grande   # ...pero con el LLM de 1.5B (si no alcanza la RAM)
    python scripts/entrenar.py --desde-cero       # descarta lo pausado y empieza de nuevo
El transformer y el LLM se cortan entre dos pasos de optimización. Guardan pesos, optimizador, plan de tasa y
azar en ml/checkpoints/ y siguen exactamente desde ahí. Los pasos cortos (bayes, modelo, evaluar, gestos) no se
cortan a la mitad: terminan y el entrenamiento frena antes del siguiente. Los pasos largos guardan además
cada hora, así que un Ctrl+C o un corte de luz pierde como mucho una hora.

Los pedidos se entrenan con ml/intencion/frases_entrenamiento.csv y se validan con eval/frases_*_validacion.csv, que
están en el repositorio. Los gestos se entrenan con los landmarks de HaGRID de ml/data/hagrid/hagrid_*.csv.xz, también
versionados. Mientras corre, le pide a Windows que no suspenda la PC. Las bases (BETO, Qwen, E5) se descargan de Hugging Face la primera vez.

Usa el mismo entorno que scripts/levantar.py (backend/.venv) y le suma ml/requirements-entrenamiento.txt; solo
reinstala si cambia alguno de los dos. La salida de cada paso queda también en ml/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from levantar import ENTORNO, PYTHON_ENTORNO, RAIZ, instalar, preparar_entorno

sys.path.append(str(RAIZ / "ml"))
import pausa  # noqa: E402  (no importa torch: el orquestador corre con cualquier Python)

ESTADO = pausa.CARPETA / "entrenamiento.json"   # pasos pendientes y opciones de la corrida pausada
EXTRA = RAIZ / "ml" / "requirements-entrenamiento.txt"
HUELLA_EXTRA = ENTORNO / "requisitos_entrenamiento.sha256"
LOGS = RAIZ / "ml" / "logs"
GESTOS = RAIZ / "ml" / "data" / "hagrid"


@dataclass
class Paso:
    nombre: str
    que: str
    comando: list
    minutos: int                  # en CPU, con todo el entrenamiento
    minutos_prueba: int | None    # con --prueba; None: no corre en la prueba
    rapido: bool = False          # acepta --rapido (versión mínima que no exporta nada)
    gestos: bool = False          # necesita los landmarks de HaGRID
    por_defecto: bool = True
    pausable: bool = False        # se puede cortar a la mitad y retomar (ml/pausa.py)
    salida: list = field(default_factory=list)


def pasos(llm):
    return [
        # Rehacer los datos (no van por defecto: las frases ya están en el repositorio)
        Paso("voz", "Frases de entrenamiento dichas con voces sintéticas y transcritas por Whisper",
             ["ml/intencion/aumentar_voz.py"], 210, None, por_defecto=False, salida=["ml/intencion/frases_voz.csv"]),
        Paso("frases", "Frases de entrenamiento (plantillas ×5, errores de tipeo y las de voz)",
             ["ml/intencion/generar_frases.py"], 3, None, por_defecto=False,
             salida=["ml/intencion/frases_entrenamiento.csv"]),
        Paso("bayes", "Naive Bayes de pedidos + calibración", ["ml/intencion/entrenar_bayes.py"], 1, 1,
             salida=["backend/modelos/bayes.json"]),
        Paso("modelo", "Intérprete E5 + regresión logística + etiquetador", ["ml/intencion/entrenar_intencion.py"],
             25, 25, salida=["backend/modelos/intencion.json"]),
        Paso("transformer", "Transformer XLM-RoBERTa large (560 M; acción + objeto BIO con Viterbi)",
             ["ml/intencion_transformer/entrenar_transformer.py", "--candidatos", "xlmr"], 960, 20, rapido=True,
             pausable=True, salida=["backend/modelos/transformer/"]),
        Paso("llm", f"LLM propio Qwen2.5 {'7B' if llm == 'maximo' else '1.5B' if llm == 'grande' else '0.5B'} + LoRA r=64",
             ["ml/llm_propio/entrenar_llm.py", "--tamano", llm],
             2400 if llm == "maximo" else 480 if llm == "grande" else 160,
             30 if llm == "maximo" else 15 if llm == "grande" else 6, rapido=True, pausable=True,
             salida=[f"backend/modelos/llm_propio_{llm}/" if llm != "chico" else "backend/modelos/llm_propio/"]),
        Paso("consenso", "Pesos del consenso (L-BFGS contra genético y voto)", ["ml/consenso/ajustar_consenso.py"],
             15, None, salida=["backend/modelos/consenso.json"]),
        Paso("evaluar", "Evaluación de punta a punta (set de prueba, todos los motores)",
             ["eval/eval_intent.py", "--frases", "prueba"], 5, None, salida=["eval/resultados/"]),
        Paso("red", "Red de gestos (MLP, cinco tamaños)", ["ml/entrenar_red_gestos.py"], 10, 3, rapido=True,
             gestos=True, salida=["frontend/ml/mlp_gestos.json"]),
        Paso("arboles", "Árbol, bosque y boosting de gestos", ["ml/arboles/entrenar_arboles.py"], 5, None,
             gestos=True, salida=["frontend/ml/arbol_gestos.json"]),
        Paso("neuroevolucion", "Algoritmo genético para la red de gestos", ["ml/evolucion/neuroevolucion.py"],
             150, 5, rapido=True, gestos=True, por_defecto=False, salida=["frontend/ml/mlp_gestos.json (si gana)"]),
    ]


def no_suspender():
    """Pide a Windows que no suspenda la PC mientras este proceso viva (el entrenamiento tarda horas). No cambia la
    configuración de energía: el pedido se termina solo cuando termina el proceso."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def hay_gestos():
    return any(GESTOS.glob("*.csv")) or any(GESTOS.glob("*.csv.xz"))


def motivo_para_saltear(paso):
    if paso.gestos and not hay_gestos():
        return "faltan los landmarks de HaGRID en ml/data/hagrid/ (¿se bajó bien el repositorio?)"
    if paso.nombre == "red" and not shutil.which("node"):
        return "hace falta Node.js (compara la red contra las reglas del visor)"
    return None


def memoria_gb():
    try:
        if os.name == "nt":
            import ctypes

            class Estado(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("total", "libre", "pag", "pag_libre", "virt",
                                                              "virt_libre", "ext")]
            e = Estado()
            e.dwLength = ctypes.sizeof(Estado)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(e))
            return e.total / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, OSError, ValueError):
        return None


def duracion(minutos):
    return f"~{minutos} min" if minutos < 90 else f"~{minutos / 60:.1f} h".replace(".0 h", " h").replace(".", ",")


def cambios_git():
    """Archivos versionados modificados (vacío si no hay git)."""
    if not shutil.which("git"):
        return set()
    salida = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=RAIZ,
                            capture_output=True, text=True).stdout
    return {linea[3:] for linea in salida.splitlines()}


RETOMAR = "python scripts/entrenar.py --continuar"
paso_actual = None   # el paso que está corriendo (para el aviso de la tecla P)


def guardar_estado(pendientes, args, llm):
    """Anota qué falta y con qué opciones, para que --continuar siga igual."""
    pausa.escribir_json(ESTADO, {"pendientes": [p.nombre for p in pendientes], "prueba": args.prueba, "llm": llm,
                                 "fecha": datetime.now().isoformat(timespec="seconds")})


def aviso_pausa():
    if paso_actual is None or paso_actual.pausable:
        return ("termina el paso de optimización en curso, guarda el punto de control y frena "
                "(guardar puede tardar un par de minutos)")
    return f"«{paso_actual.nombre}» no se corta a la mitad: cuando termine, frena antes del paso siguiente"


def alternar_pausa():
    if pausa.pedida():
        pausa.cancelar()
        print("\n>>> Pausa cancelada: sigue entrenando.\n", flush=True)
    else:
        pausa.pedir("tecla")
        print(f"\n>>> Pausa pedida: {aviso_pausa()}. Apretá P otra vez para cancelarla.\n", flush=True)


def escuchar_pausa(horas):
    """Hilo aparte: la tecla P pide (o cancela) la pausa; con --horas, se pide sola al cumplirse el tiempo."""
    limite = time.monotonic() + horas * 3600 if horas else None
    if os.name == "nt":
        import msvcrt
    else:
        import select
    while True:
        if limite and time.monotonic() >= limite:
            limite = None
            if not pausa.pedida():
                pausa.pedir("horas")
                print(f"\n>>> Pasaron {horas:g} h: {aviso_pausa()}. Apretá P para seguir igual.\n", flush=True)
        try:
            if os.name == "nt":
                while msvcrt.kbhit():
                    tecla = msvcrt.getwch()
                    if tecla in ("\x00", "\xe0"):   # flechas y teclas de función mandan dos códigos
                        msvcrt.getwch()
                    elif tecla.lower() == "p":
                        alternar_pausa()
            elif sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "p":   # en Linux/macOS: P y Enter
                    alternar_pausa()
        except (OSError, ValueError):   # sin consola (salida redirigida): queda --pausar y --horas
            pass
        time.sleep(0.2)


def correr(paso, prueba, log):
    """Corre el paso con el Python del entorno; muestra la salida y la copia al log."""
    comando = [str(PYTHON_ENTORNO), *paso.comando] + (["--rapido"] if prueba and paso.rapido else [])
    entorno = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
               # La evaluación no usa Claude aunque haya clave en .env: entrenar no tiene que gastar crédito
               "ANTHROPIC_API_KEY": ""}
    with open(log, "w", encoding="utf-8") as archivo:
        archivo.write(" ".join(comando) + "\n\n")
        proceso = subprocess.Popen(comando, cwd=RAIZ, env=entorno, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for linea in proceso.stdout:
                print(f"  {linea}", end="", flush=True)
                archivo.write(linea)
            return proceso.wait()
        except KeyboardInterrupt:
            proceso.terminate()
            proceso.wait()
            raise


def main():
    ap = argparse.ArgumentParser(description="Entrena los modelos del proyecto con los datos del repositorio.")
    ap.add_argument("--listar", action="store_true", help="mostrar los pasos y salir")
    ap.add_argument("--prueba", action="store_true",
                    help="versión mínima de cada paso, para ver que la máquina puede entrenar (~20 min)")
    ap.add_argument("--llm", choices=["maximo", "grande", "chico"],
                    help="tamaño del LLM propio (por defecto: maximo 7B; con --prueba, chico)")
    ap.add_argument("--solo", help="pasos separados por coma (ej. bayes,modelo)")
    ap.add_argument("--desde", help="empezar en este paso y seguir con los demás (para retomar)")
    ap.add_argument("--con-voz", action="store_true",
                    help="antes de entrenar, rehacer las frases con voz de Whisper (~3,5 h más)")
    ap.add_argument("--continuar", action="store_true", help="retomar el entrenamiento pausado, con sus opciones")
    ap.add_argument("--desde-cero", action="store_true",
                    help="descartar el entrenamiento pausado y sus puntos de control, y empezar de nuevo")
    ap.add_argument("--horas", type=float, help="pausar solo después de tantas horas (ej. 8)")
    ap.add_argument("--pausar", action="store_true",
                    help="pedir la pausa al entrenamiento que corre en otra terminal y salir")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.pausar:
        pausa.pedir("otra terminal")
        print("Pausa pedida: el entrenamiento termina el paso en curso, guarda y frena.\n"
              f"Para retomarlo después: {RETOMAR}")
        return
    if sys.version_info < (3, 10) or struct.calcsize("P") != 8:
        sys.exit("Hace falta Python 3.10 o más nuevo, de 64 bits (PyTorch no tiene versión de 32 bits).")

    pendientes = None
    if args.continuar:
        if args.desde_cero or args.solo or args.desde or args.prueba or args.con_voz:
            sys.exit("--continuar usa las opciones de la corrida pausada: solo se combina con --horas y --llm.")
        if not ESTADO.exists():
            sys.exit("No hay ningún entrenamiento pausado para continuar.")
        estado = json.loads(ESTADO.read_text(encoding="utf-8"))
        pendientes, args.prueba = estado["pendientes"], estado["prueba"]
        print(f"Continuando el entrenamiento pausado el {estado['fecha'].replace('T', ' a las ')}: "
              f"falta {', '.join(pendientes)}.")
        # Cambiar el tamaño del LLM al retomar (por ejemplo, si no alcanza la RAM para el 7B). Los pasos ya hechos
        # no se tocan; el LLM arranca de cero con la base nueva: su punto de control lleva la huella de la base.
        if args.llm and args.llm != estado["llm"]:
            if "llm" not in pendientes:
                sys.exit(f"El LLM ({estado['llm']}) ya se entrenó en esta corrida: --llm no cambia nada. "
                         f"Para entrenarlo con otra base: python scripts/entrenar.py --solo llm --llm {args.llm}")
            anterior = pausa.CARPETA / f"llm_{estado['llm']}{'_prueba' if args.prueba else ''}.pt"
            print(f"El LLM pasa de «{estado['llm']}» a «{args.llm}».")
            if anterior.exists():
                print(f"  Queda {anterior.relative_to(RAIZ).as_posix()} ({anterior.stat().st_size / 2**30:.1f} GB), "
                      f"el avance del LLM «{estado['llm']}». Si no vas a volver a esa base, se puede borrar.")
        else:
            args.llm = estado["llm"]
    elif args.desde_cero:
        shutil.rmtree(pausa.CARPETA, ignore_errors=True)
        print("Se descartó el entrenamiento pausado y sus puntos de control (ml/checkpoints/).")
    elif ESTADO.exists() and not args.listar:
        estado = json.loads(ESTADO.read_text(encoding="utf-8"))
        sys.exit(f"Hay un entrenamiento pausado el {estado['fecha'].replace('T', ' a las ')} "
                 f"(falta {', '.join(estado['pendientes'])}).\n"
                 f"  Para seguirlo:          {RETOMAR}\n"
                 "  Para descartarlo:       python scripts/entrenar.py --desde-cero [otras opciones]")
    llm = args.llm or ("chico" if args.prueba else "maximo")
    todos = pasos(llm)
    nombres = [p.nombre for p in todos]

    if args.listar:
        print(f"{'paso':15s} {'tarda':>8s} {'prueba':>8s}  qué entrena")
        for p in todos:
            motivo = motivo_para_saltear(p)
            opcional = "  [con --con-voz]" if p.nombre in ("voz", "frases") else "  [solo con --solo]"
            extra = f"  [no se puede: {motivo}]" if motivo else ("" if p.por_defecto else opcional)
            prueba = duracion(p.minutos_prueba) if p.minutos_prueba else "-"
            print(f"{p.nombre:15s} {duracion(p.minutos):>8s} {prueba:>8s}  {p.que}{extra}")
        return

    for valor in filter(None, [args.desde, *(args.solo or "").split(",")]):
        if valor.strip() not in nombres:
            sys.exit(f"No existe el paso «{valor}». Pasos: {', '.join(nombres)}")
    if pendientes is not None:
        elegidos = [p for p in todos if p.nombre in pendientes]
    elif args.solo:
        pedidos = {s.strip() for s in args.solo.split(",")}
        elegidos = [p for p in todos if p.nombre in pedidos]
    else:
        extra = {"voz", "frases"} if args.con_voz else set()
        elegidos = [p for p in todos if p.por_defecto or p.nombre == args.desde or p.nombre in extra]
        if args.desde:
            elegidos = elegidos[[p.nombre for p in elegidos].index(args.desde):]
    if args.prueba:
        elegidos = [p for p in elegidos if p.minutos_prueba]

    saltear = {p.nombre: m for p in elegidos if (m := motivo_para_saltear(p))}
    a_correr = [p for p in elegidos if p.nombre not in saltear]
    if not a_correr:
        sys.exit("No queda ningún paso para correr: " + "; ".join(f"{n}: {m}" for n, m in saltear.items()))

    # Avisos antes de arrancar algo largo
    libre = shutil.disk_usage(RAIZ).free / 2**30
    if libre < 15:
        print(f"Atención: quedan {libre:.0f} GB libres. Las descargas y los modelos ocupan unos 10-15 GB.")
    memoria = memoria_gb()
    minimo_llm = 64 if llm == "maximo" else 16
    if memoria and memoria < minimo_llm and any(p.nombre == "llm" for p in a_correr):
        alternativa = "--llm grande" if llm == "maximo" else "--llm chico"
        print(f"Atención: el equipo tiene {memoria:.0f} GB de RAM; Qwen {llm} puede quedarse sin memoria. "
              f"Para reducir el uso de RAM, usá {alternativa}.")

    preparar_entorno()
    instalar(EXTRA, HUELLA_EXTRA, "las dependencias de entrenamiento")
    LOGS.mkdir(parents=True, exist_ok=True)
    no_suspender()

    total = sum(p.minutos_prueba if args.prueba else p.minutos for p in a_correr)
    print(f"\n{'Prueba' if args.prueba else 'Entrenamiento'}: {len(a_correr)} pasos, {duracion(total)} en CPU"
          " (más las descargas de la primera vez).")
    for nombre, motivo in saltear.items():
        print(f"  se saltea {nombre}: {motivo}")

    global paso_actual
    pausa.cancelar()   # un pedido viejo (de --pausar sin nada corriendo) no tiene que frenar esta corrida
    teclado = "Apretá P para pausar" if sys.stdin.isatty() else "Para pausar: python scripts/entrenar.py --pausar"
    print(f"{teclado}: termina lo que está haciendo, guarda y frena; después, {RETOMAR}"
          + (f". Se pausa solo a las {args.horas:g} h." if args.horas else "."))
    threading.Thread(target=escuchar_pausa, args=(args.horas,), daemon=True).start()

    antes = cambios_git()
    resumen, pausado = [], None
    for i, paso in enumerate(a_correr, 1):
        if pausa.pedida():   # la pausa llegó cuando ya terminaba el paso anterior
            pausado = a_correr[i - 1:]
            break
        paso_actual = paso
        print(f"\n[{i}/{len(a_correr)}] {paso.nombre}: {paso.que} "
              f"({duracion(paso.minutos_prueba if args.prueba else paso.minutos)})", flush=True)
        log = LOGS / f"{datetime.now():%Y-%m-%d_%H%M}_{paso.nombre}{'_prueba' if args.prueba else ''}.log"
        t0 = time.perf_counter()
        try:
            codigo = correr(paso, args.prueba, log)
        except KeyboardInterrupt:
            guardar_estado(a_correr[i - 1:], args, llm)
            ultimo = " (desde el último punto de control guardado)" if paso.pausable else ""
            sys.exit(f"\nCortado en «{paso.nombre}». Para retomar{ultimo}: {RETOMAR}")
        minutos = (time.perf_counter() - t0) / 60
        if codigo == pausa.CODIGO:   # el paso guardó su punto de control y salió a propósito
            resumen.append((paso.nombre, "pausa", minutos))
            pausado = a_correr[i - 1:]
            break
        resumen.append((paso.nombre, "ok   " if codigo == 0 else "FALLÓ", minutos))
        if codigo != 0:
            guardar_estado(a_correr[i - 1:], args, llm)
            print(f"\nFalló «{paso.nombre}» (código {codigo}). La salida completa está en "
                  f"{log.relative_to(RAIZ).as_posix()}.\nCuando lo arregles: {RETOMAR}")
            break
    paso_actual = None

    print("\nResumen:")
    for nombre, como, minutos in resumen:
        print(f"  {como} {nombre:15s} {minutos:6.1f} min")
    if pausado:
        guardar_estado(pausado, args, llm)
        pausa.cancelar()
        print(f"\nEntrenamiento en pausa. Falta: {', '.join(p.nombre for p in pausado)}.\n"
              f"Cuando quieras seguir: {RETOMAR}")
        return
    if all(c == "ok   " for _, c, _ in resumen) and len(resumen) == len(a_correr):
        ESTADO.unlink(missing_ok=True)
        pausa.cancelar()
        if args.prueba:
            print("La máquina puede entrenar todo. Para el entrenamiento completo: python scripts/entrenar.py")
        else:
            print("Listo. Los reportes están en ml/resultados/ y la comparación de motores en eval/resultados/.\n"
                  "Para probarlos: python scripts/levantar.py")
        cambios = cambios_git() - antes
        if cambios:
            print("\nArchivos versionados que cambió el entrenamiento (revisalos antes de commitear):\n  "
                  + "\n  ".join(sorted(cambios)))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
