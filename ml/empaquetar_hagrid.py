"""Une los landmarks extraídos de varias muestras de HaGRID (ml/extraer_hagrid.py) sin repetir fotos y los guarda
comprimidos en ml/data/hagrid/, partidos en archivos de menos de 95 MB (GitHub rechaza archivos de más de 100 MB).

Las muestras de Hugging Face se superponen: una foto que ya está en la primera se saca de las siguientes (por la
columna "imagen", el identificador de HaGRID).

Uso (desde la raíz):
    backend/.venv/Scripts/python ml/empaquetar_hagrid.py ml/data/externo/hagrid.csv ml/data/externo/hagrid500k.csv
"""
import argparse
import lzma
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar import DATOS_HAGRID  # noqa: E402

FILAS_POR_PARTE = 120_000   # ~70 MB sin comprimir, ~20 MB con xz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    args = ap.parse_args()

    partes, vistas = [], set()
    for ruta in args.csvs:
        df = pd.read_csv(ruta, dtype={"persona": str, "mano": str})
        nuevas = df[~df["imagen"].isin(vistas)]
        print(f"{ruta}: {len(df):,} manos · {len(nuevas):,} de fotos nuevas")
        vistas |= set(df["imagen"])
        partes.append(nuevas)
    todo = pd.concat(partes, ignore_index=True)
    print(f"total: {len(todo):,} manos de {todo['imagen'].nunique():,} fotos y {todo['persona'].nunique():,} personas")

    for viejo in [*DATOS_HAGRID.glob("*.csv.xz"), *DATOS_HAGRID.glob("*.csv")]:
        viejo.unlink()
    for i, desde in enumerate(range(0, len(todo), FILAS_POR_PARTE), 1):
        ruta = DATOS_HAGRID / f"hagrid_{i:02d}.csv.xz"
        with lzma.open(ruta, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
            todo.iloc[desde:desde + FILAS_POR_PARTE].to_csv(f, index=False, lineterminator="\n")
        tamano = ruta.stat().st_size / 2**20
        assert tamano < 95, f"{ruta.name} pesa {tamano:.0f} MB: bajar FILAS_POR_PARTE"
        print(f"  {ruta.name}: {tamano:.0f} MB")


if __name__ == "__main__":
    main()
