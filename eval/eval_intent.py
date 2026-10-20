"""Mide los aciertos del LLM de intención contra frases_prueba.csv (meta: >= 85 %).

Uso (desde la raíz):  python eval/eval_intent.py
"""
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from intent import interpretar  # noqa: E402
from safety import validar_intencion  # noqa: E402

CAMPOS = ("accion", "objeto", "parametros")


def coincide(obtenido: dict, esperado: dict) -> bool:
    return all(obtenido.get(c) == esperado.get(c) for c in CAMPOS)


def main():
    with open(Path(__file__).parent / "frases_prueba.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    aciertos = 0
    for fila in filas:
        obtenido = validar_intencion(interpretar(fila["frase"]))
        ok = coincide(obtenido, json.loads(fila["json_esperado"]))
        aciertos += ok
        if not ok:
            print(f"FALLA: {fila['frase']!r} -> {obtenido}")

    print(f"\nAciertos: {aciertos}/{len(filas)} ({100 * aciertos / len(filas):.0f} %)")


if __name__ == "__main__":
    main()
