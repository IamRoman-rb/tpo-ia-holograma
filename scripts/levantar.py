"""Levanta el backend y el frontend juntos, en una sola terminal.

Uso (desde la raíz del repositorio, con cualquier Python 3.10+):
    python scripts/levantar.py                  # backend en :8000, frontend en :5500 y abre el navegador
    python scripts/levantar.py --sin-navegador
    python scripts/levantar.py --sin-voz        # no carga Whisper (arranca más rápido y usa ~3 GB menos)
    python scripts/levantar.py --recargar       # reinicia el backend al guardar cambios en backend/

La primera vez crea backend/.venv e instala backend/requirements.txt. Después solo vuelve a instalar si
requirements.txt cambió (guarda su huella en backend/.venv/requisitos.sha256). Ctrl+C apaga los dos.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BACKEND = RAIZ / "backend"
ENTORNO = BACKEND / ".venv"
REQUISITOS = BACKEND / "requirements.txt"
HUELLA = ENTORNO / "requisitos.sha256"
PYTHON_ENTORNO = ENTORNO / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def preparar_entorno():
    if not PYTHON_ENTORNO.exists():
        print(f"Creando el entorno de Python en {ENTORNO.relative_to(RAIZ)} (solo la primera vez)…")
        venv.create(ENTORNO, with_pip=True)
    instalar(REQUISITOS, HUELLA, "las dependencias del backend")


def instalar(requisitos, huella_guardada, que):
    """pip install -r requisitos en el entorno, solo si el archivo cambió desde la última vez."""
    huella = hashlib.sha256(requisitos.read_bytes()).hexdigest()
    if huella_guardada.exists() and huella_guardada.read_text().strip() == huella:
        return
    print(f"Instalando {que} (solo cuando cambia {requisitos.relative_to(RAIZ).as_posix()})…")
    subprocess.run([str(PYTHON_ENTORNO), "-m", "pip", "install", "-q", "-r", str(requisitos)], check=True)
    huella_guardada.write_text(huella)


def puerto_ocupado(puerto):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", puerto)) == 0


def reenviar(proceso, etiqueta, ocultar=()):
    """Copia la salida del proceso a esta terminal, con una etiqueta delante de cada línea.
    Lee siempre hasta el final: si se dejara de leer, el proceso se trabaría al llenarse la tubería."""
    for linea in proceso.stdout:
        if linea.strip() and not any(o in linea for o in ocultar):
            print(f"{etiqueta:7s} {linea}", end="", flush=True)


def esperar_backend(puerto, abrir):
    """Espera a que el backend responda y a que carguen los motores de pedidos (el LLM propio tarda ~1 min)."""
    url = f"http://localhost:{puerto}/health"
    avisado = False
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.loads(r.read()).get("motores_listos", True):
                    break
            if not avisado:
                print("\n  El backend responde; cargando los motores de pedidos (la primera vez ~1 min)…", flush=True)
                avisado = True
        except (OSError, ValueError):
            pass
        time.sleep(2)
    print(f"\n  Backend listo: {url}", flush=True)
    if abrir:
        webbrowser.open(abrir)


def main():
    ap = argparse.ArgumentParser(description="Levanta el backend y el frontend en una sola terminal.")
    ap.add_argument("--puerto-backend", type=int, default=8000)
    ap.add_argument("--puerto-frontend", type=int, default=5500)
    ap.add_argument("--sin-navegador", action="store_true")
    ap.add_argument("--sin-voz", action="store_true", help="no cargar Whisper (VOZ_MODO=desactivado)")
    ap.add_argument("--recargar", action="store_true", help="reiniciar el backend al cambiar el código")
    args = ap.parse_args()

    # La consola de Windows puede no tener todos los caracteres que escriben los modelos (barras, flechas)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.version_info < (3, 10):
        sys.exit("Hace falta Python 3.10 o más nuevo.")
    for puerto in (args.puerto_backend, args.puerto_frontend):
        if puerto_ocupado(puerto):
            sys.exit(f"El puerto {puerto} ya está en uso: ¿quedó el proyecto levantado en otra terminal?")
    preparar_entorno()

    entorno = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if args.sin_voz:
        entorno["VOZ_MODO"] = "desactivado"
    comun = {"cwd": RAIZ, "env": entorno, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
             "text": True, "encoding": "utf-8", "errors": "replace"}
    backend = subprocess.Popen([str(PYTHON_ENTORNO), "-m", "uvicorn", "main:app", "--app-dir", "backend",
                                "--port", str(args.puerto_backend)]
                               + (["--reload", "--reload-dir", "backend"] if args.recargar else []), **comun)
    frontend = subprocess.Popen([sys.executable, "-m", "http.server", str(args.puerto_frontend),
                                 "--directory", "frontend"], **comun)
    procesos = {"backend": backend, "frontend": frontend}

    threading.Thread(target=reenviar, args=(backend, "[back]", ("Loading weights", "INFO httpx", "HF_TOKEN")), daemon=True).start()
    # El servidor estático anota cada archivo que sirve: solo se muestran los errores
    threading.Thread(target=reenviar, args=(frontend, "[front]", ('" 200 ', '" 304 ', "Serving HTTP")),
                     daemon=True).start()
    visor = f"http://localhost:{args.puerto_frontend}"
    threading.Thread(target=esperar_backend, args=(args.puerto_backend, None if args.sin_navegador else visor),
                     daemon=True).start()
    print(f"\n  Visor:        {visor}\n  Red neuronal: {visor}/red.html\n  LLM:          {visor}/llm.html\n"
          f"  Backend:      http://localhost:{args.puerto_backend} (cargando modelos…)\n"
          "  Ctrl+C apaga los dos.\n", flush=True)

    try:
        while all(p.poll() is None for p in procesos.values()):
            time.sleep(0.5)
        caido = next(n for n, p in procesos.items() if p.poll() is not None)
        print(f"\nEl {caido} se cerró (código {procesos[caido].returncode}); apago el resto.")
    except KeyboardInterrupt:
        print("\nApagando…")
    finally:
        for p in procesos.values():
            if p.poll() is None:
                p.terminate()
        for p in procesos.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
