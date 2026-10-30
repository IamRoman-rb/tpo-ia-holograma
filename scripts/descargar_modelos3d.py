"""Baja los modelos 3D del catálogo (los .glb de Objaverse elegidos en ml/catalogo3d/seleccion.csv).

Uso (desde la raíz, con cualquier Python 3.10+):
    python scripts/descargar_modelos3d.py            # todos (~50 GB)
    python scripts/descargar_modelos3d.py --gb 10    # los mejores de cada categoría hasta 10 GB

Van a HOLOGRAMA_DATOS/objaverse/glbs (por defecto ml/data/externo/, que no se versiona). Para usar otro disco:
HOLOGRAMA_DATOS=D:/holograma. Se puede cortar y volver a correr: solo baja lo que falta. El backend muestra los
modelos que estén bajados; los demás no aparecen en la búsqueda.
"""
import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "catalogo3d"))
from comun import GLBS, HF, SELECCION, descargar  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Baja los modelos 3D del catálogo.")
    ap.add_argument("--gb", type=float, help="tope en GB (por defecto, todos)")
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(SELECCION, encoding="utf-8") as f:
        filas = sorted(csv.DictReader(f), key=lambda x: (int(x["ronda"]), -float(x["calidad"])))
    if args.gb:   # las primeras rondas tienen el mejor modelo de cada categoría
        tope, acumulado, recorte = args.gb * 2**30, 0, []
        for fila in filas:
            acumulado += int(fila["tamano"])
            if acumulado > tope:
                break
            recorte.append(fila)
        filas = recorte
    faltan = [x for x in filas if not (GLBS / f"{x['uid']}.glb").exists()]
    total = sum(int(x["tamano"]) for x in faltan)
    libre = shutil.disk_usage(GLBS.parent.parent if GLBS.parent.parent.exists() else Path.cwd()).free
    print(f"{len(filas):,} modelos en la selección · faltan {len(faltan):,} ({total / 2**30:.1f} GB) · "
          f"libres {libre / 2**30:.0f} GB · destino {GLBS}")
    if total > libre - 5 * 2**30:
        sys.exit("No alcanza el disco (se dejan 5 GB libres). Usá --gb o HOLOGRAMA_DATOS en otro disco.")

    base = HF + "/allenai/objaverse/resolve/main/"
    t0, bajado, errores = time.time(), 0, []
    with ThreadPoolExecutor(args.hilos) as ex:
        futuros = {ex.submit(descargar, base + x["ruta"], GLBS / f"{x['uid']}.glb"): x for x in faltan}
        for i, futuro in enumerate(as_completed(futuros), 1):
            try:
                bajado += futuro.result()
            except Exception as e:   # uno que falla no frena el resto; se reintenta en la próxima corrida
                errores.append((futuros[futuro]["uid"], type(e).__name__))
            if i % 100 == 0 or i == len(faltan):
                velocidad = bajado / 2**20 / max(time.time() - t0, 1)
                resta = (total - bajado) / 2**20 / max(velocidad, 0.01) / 60
                print(f"  {i:,}/{len(faltan):,} · {bajado / 2**30:.1f} GB · {velocidad:.1f} MB/s · faltan ~{resta:.0f} min",
                      flush=True)
    print(f"Listo: {len(faltan) - len(errores):,} modelos nuevos." +
          (f" {len(errores)} fallaron (volvé a correr el script): {errores[:5]}" if errores else ""))


if __name__ == "__main__":
    main()
