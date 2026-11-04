"""Pausa ordenada del entrenamiento: frenar sin romper nada y retomar después desde el mismo punto.

scripts/entrenar.py crea ml/checkpoints/PAUSAR cuando se aprieta la tecla P, cuando se cumple --horas o cuando
otra terminal corre `python scripts/entrenar.py --pausar`. Los entrenamientos largos (transformer y LLM propio)
miran ese archivo al terminar cada paso de optimización: si está, guardan su punto de control y salen con el
código CODIGO. La vez siguiente arrancan desde ese paso, con los mismos pesos, el mismo optimizador, el mismo
plan de tasa y el mismo estado del azar.

Además guardan solos cada AUTOGUARDADO_MIN minutos. Así, si se corta la luz o se aprieta Ctrl+C, se pierde
como mucho ese rato de trabajo.

Cada punto de control se escribe primero en un archivo temporal y después se renombra. El renombre (os.replace)
es atómico: si algo falla mientras se guarda, queda entero el punto anterior.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "ml" / "checkpoints"
SENAL = CARPETA / "PAUSAR"
CODIGO = 75             # código de salida: «frené a propósito y se puede retomar» (EX_TEMPFAIL)
AUTOGUARDADO_MIN = 60


def pedida():
    return SENAL.exists()


def pedir(motivo="tecla"):
    CARPETA.mkdir(parents=True, exist_ok=True)
    SENAL.write_text(motivo, encoding="utf-8")


def cancelar():
    SENAL.unlink(missing_ok=True)


def salir():
    print("\n>>> Pausa: el punto de control quedó guardado. Se retoma con: python scripts/entrenar.py --continuar",
          flush=True)
    sys.exit(CODIGO)


def huella(*partes):
    """Resumen de lo que define una corrida (base, hiperparámetros, datos). Si cambia algo, el punto de
    control guardado no sirve y se empieza de cero."""
    texto = json.dumps(partes, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def escribir_json(ruta, datos):
    """JSON escrito de forma atómica (temporal + renombre)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporal, ruta)


class PuntoDeControl:
    """El archivo .pt de un entrenamiento: cargar al empezar, guardar al pausar o cada AUTOGUARDADO_MIN minutos,
    borrar al terminar."""

    def __init__(self, nombre, huella_corrida):
        self.ruta = CARPETA / f"{nombre}.pt"
        self.huella = huella_corrida
        self.ultimo = time.monotonic()

    def cargar(self):
        """El estado guardado si corresponde a esta misma corrida; si no, None."""
        if not self.ruta.exists():
            return None
        import torch
        try:
            estado = torch.load(self.ruta, map_location="cpu", weights_only=True)
        except Exception as e:  # archivo de otra versión de torch, disco lleno al guardar, etc.
            print(f"  (no se pudo leer {self.ruta.name}: {e}; se empieza de cero)", flush=True)
            return None
        if estado.get("huella") != self.huella:
            print(f"  ({self.ruta.name} es de otra configuración o de otros datos; se empieza de cero)", flush=True)
            return None
        return estado

    def toca_autoguardar(self):
        return time.monotonic() - self.ultimo >= AUTOGUARDADO_MIN * 60

    def guardar(self, estado):
        import torch
        t0 = time.perf_counter()
        CARPETA.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta.with_name(self.ruta.name + ".tmp")
        torch.save({**estado, "huella": self.huella}, temporal)
        os.replace(temporal, self.ruta)
        self.ultimo = time.monotonic()
        tamano = self.ruta.stat().st_size / 2**30
        print(f"  punto de control guardado: paso {estado.get('paso')} -> ml/checkpoints/{self.ruta.name} "
              f"({tamano:.1f} GB, {time.perf_counter() - t0:.0f} s)", flush=True)

    def borrar(self):
        self.ruta.unlink(missing_ok=True)
        self.ruta.with_name(self.ruta.name + ".tmp").unlink(missing_ok=True)
