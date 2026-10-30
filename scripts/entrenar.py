"""Entrena los modelos del proyecto con los datos del repositorio, en orden y en una sola terminal.

Uso (desde la raíz del repositorio, con cualquier Python 3.10+ de 64 bits):
    python scripts/entrenar.py --listar            # qué pasos hay, cuánto tarda cada uno y cuáles se pueden correr
    python scripts/entrenar.py --prueba            # ~20 min: cada paso en su versión mínima, sin pisar los modelos lentos
    python scripts/entrenar.py                     # todo (~13 h en CPU con el LLM grande)
    python scripts/entrenar.py --llm chico         # con el LLM de 0.5B (~2,5 h en vez de ~8 h)
    python scripts/entrenar.py --solo bayes,modelo # solo esos pasos
    python scripts/entrenar.py --desde llm         # retomar después de un corte

Los pedidos se entrenan con ml/intencion/frases_entrenamiento.csv y se validan con eval/frases_*_validacion.csv, que
están en el repositorio. Los gestos se entrenan con los landmarks de HaGRID de ml/data/hagrid/hagrid.csv.xz, también
versionados. Las bases (BETO, Qwen, E5) se descargan de Hugging Face la primera vez.

Usa el mismo entorno que scripts/levantar.py (backend/.venv) y le suma ml/requirements-entrenamiento.txt; solo
reinstala si cambia alguno de los dos. La salida de cada paso queda también en ml/logs/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from levantar import ENTORNO, PYTHON_ENTORNO, RAIZ, instalar, preparar_entorno

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
    salida: list = field(default_factory=list)


def pasos(llm):
    return [
        Paso("bayes", "Naive Bayes de pedidos + calibración", ["ml/intencion/entrenar_bayes.py"], 1, 1,
             salida=["backend/modelos/bayes.json"]),
        Paso("modelo", "Intérprete E5 + regresión logística + etiquetador", ["ml/intencion/entrenar_intencion.py"],
             25, 25, salida=["backend/modelos/intencion.json"]),
        Paso("transformer", "Transformer BETO (acción + objeto BIO con Viterbi)",
             ["ml/intencion_transformer/entrenar_transformer.py", "--candidatos", "beto"], 240, 4, rapido=True,
             salida=["backend/modelos/transformer/"]),
        Paso("llm", f"LLM propio Qwen2.5 {'1.5B' if llm == 'grande' else '0.5B'} + LoRA",
             ["ml/llm_propio/entrenar_llm.py", "--tamano", llm], 480 if llm == "grande" else 160,
             15 if llm == "grande" else 6, rapido=True,
             salida=[f"backend/modelos/{'llm_propio_grande' if llm == 'grande' else 'llm_propio'}/"]),
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


def retomar(nombre, args):
    return (f"python scripts/entrenar.py --desde {nombre}" + (" --prueba" if args.prueba else "")
            + (f" --llm {args.llm}" if args.llm else ""))


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
    ap.add_argument("--llm", choices=["grande", "chico"],
                    help="tamaño del LLM propio (por defecto: grande; con --prueba, chico)")
    ap.add_argument("--solo", help="pasos separados por coma (ej. bayes,modelo)")
    ap.add_argument("--desde", help="empezar en este paso y seguir con los demás (para retomar)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.version_info < (3, 10) or struct.calcsize("P") != 8:
        sys.exit("Hace falta Python 3.10 o más nuevo, de 64 bits (PyTorch no tiene versión de 32 bits).")
    llm = args.llm or ("chico" if args.prueba else "grande")
    todos = pasos(llm)
    nombres = [p.nombre for p in todos]

    if args.listar:
        print(f"{'paso':15s} {'tarda':>8s} {'prueba':>8s}  qué entrena")
        for p in todos:
            motivo = motivo_para_saltear(p)
            extra = f"  [no se puede: {motivo}]" if motivo else ("" if p.por_defecto else "  [solo con --solo]")
            prueba = duracion(p.minutos_prueba) if p.minutos_prueba else "-"
            print(f"{p.nombre:15s} {duracion(p.minutos):>8s} {prueba:>8s}  {p.que}{extra}")
        return

    for valor in filter(None, [args.desde, *(args.solo or "").split(",")]):
        if valor.strip() not in nombres:
            sys.exit(f"No existe el paso «{valor}». Pasos: {', '.join(nombres)}")
    if args.solo:
        pedidos = {s.strip() for s in args.solo.split(",")}
        elegidos = [p for p in todos if p.nombre in pedidos]
    else:
        elegidos = [p for p in todos if p.por_defecto or p.nombre == args.desde]
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
    if memoria and memoria < 16 and any(p.nombre == "llm" for p in a_correr) and llm == "grande":
        print(f"Atención: el equipo tiene {memoria:.0f} GB de RAM. Si el LLM grande se queda sin memoria, "
              "usá --llm chico.")

    preparar_entorno()
    instalar(EXTRA, HUELLA_EXTRA, "las dependencias de entrenamiento")
    LOGS.mkdir(parents=True, exist_ok=True)

    total = sum(p.minutos_prueba if args.prueba else p.minutos for p in a_correr)
    print(f"\n{'Prueba' if args.prueba else 'Entrenamiento'}: {len(a_correr)} pasos, {duracion(total)} en CPU"
          " (más las descargas de la primera vez).")
    for nombre, motivo in saltear.items():
        print(f"  se saltea {nombre}: {motivo}")

    antes = cambios_git()
    resumen = []
    for i, paso in enumerate(a_correr, 1):
        print(f"\n[{i}/{len(a_correr)}] {paso.nombre}: {paso.que} "
              f"({duracion(paso.minutos_prueba if args.prueba else paso.minutos)})", flush=True)
        log = LOGS / f"{datetime.now():%Y-%m-%d_%H%M}_{paso.nombre}{'_prueba' if args.prueba else ''}.log"
        t0 = time.perf_counter()
        try:
            codigo = correr(paso, args.prueba, log)
        except KeyboardInterrupt:
            sys.exit(f"\nCortado en «{paso.nombre}». Para retomar: {retomar(paso.nombre, args)}")
        minutos = (time.perf_counter() - t0) / 60
        resumen.append((paso.nombre, codigo, minutos))
        if codigo != 0:
            print(f"\nFalló «{paso.nombre}» (código {codigo}). La salida completa está en "
                  f"{log.relative_to(RAIZ).as_posix()}.\nCuando lo arregles: {retomar(paso.nombre, args)}")
            break

    print("\nResumen:")
    for nombre, codigo, minutos in resumen:
        print(f"  {'ok   ' if codigo == 0 else 'FALLÓ'} {nombre:15s} {minutos:6.1f} min")
    if all(c == 0 for _, c, _ in resumen) and len(resumen) == len(a_correr):
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
