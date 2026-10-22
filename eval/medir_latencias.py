"""Mide las latencias del pedido por etapa (sección 15: "Pedido → modelo en pantalla (catálogo) < 2 s").

Etapas: interpretar el pedido (/intent), buscar el objeto (/object) y descargar el .glb.
El tiempo de dibujar el modelo lo mide el navegador: panel de depuración del visor (tecla D),
línea "pedido_catalogo", que cubre todo el recorrido de punta a punta.

Uso (desde la raíz, con el backend y el frontend levantados):
    python eval/medir_latencias.py [--repeticiones 5]

El backend limita los pedidos por minuto (sección 16.5). Para medir, levantarlo con un límite alto:
    LIMITE_PEDIDOS_MINUTO=1000 uvicorn main:app --port 8000
"""
import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
sys.stderr.reconfigure(encoding="utf-8")

API = os.getenv("API", "http://localhost:8000")
FRONT = os.getenv("FRONT", "http://localhost:5500")
RESULTADOS = Path(__file__).parent / "resultados"


def pedidos_de_prueba():
    with open(Path(__file__).parent / "frases_prueba.csv", encoding="utf-8") as f:
        return [fila["frase"] for fila in csv.DictReader(f) if fila["accion"] == "mostrar"]


def medir(cliente, metodo, url, **kw):
    t0 = time.perf_counter()
    r = cliente.request(metodo, url, **kw)
    if r.status_code == 429:
        sys.exit("El backend rechazó pedidos por el límite de frecuencia: la medición no sería válida.\n"
                 "Levantalo con LIMITE_PEDIDOS_MINUTO=1000 (ver el encabezado de este script).")
    return r, (time.perf_counter() - t0) * 1000


def p95(valores):
    o = sorted(valores)
    return o[min(len(o) - 1, int(0.95 * len(o)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticiones", type=int, default=5)
    args = ap.parse_args()

    with httpx.Client(timeout=330) as cliente:
        try:
            salud = cliente.get(f"{API}/health").json()
        except httpx.HTTPError:
            sys.exit(f"El backend no responde en {API}")
        print(f"Backend: {salud}\n")

        filas = []
        for rep in range(args.repeticiones):
            for frase in pedidos_de_prueba():
                fila = {"repeticion": rep, "frase": frase}
                r, fila["ms_intent"] = medir(cliente, "POST", f"{API}/intent", json={"frase": frase})
                intencion = r.json()
                fila["motor"] = intencion.get("motor")
                fila["ms_object"] = fila["ms_glb"] = None
                if intencion.get("accion") == "mostrar":
                    r, fila["ms_object"] = medir(cliente, "POST", f"{API}/object", json={"frase": intencion["objeto"]})
                    if r.status_code == 200:
                        fila["origen"] = r.json().get("origen")
                        _, fila["ms_glb"] = medir(cliente, "GET", f"{FRONT}/{r.json()['modelo_url']}")
                partes = [fila[k] for k in ("ms_intent", "ms_object", "ms_glb") if fila[k] is not None]
                fila["ms_total"] = sum(partes)
                filas.append(fila)

    RESULTADOS.mkdir(exist_ok=True)
    columnas = ["repeticion", "frase", "motor", "origen", "ms_intent", "ms_object", "ms_glb", "ms_total"]
    with open(RESULTADOS / "latencias_backend.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    print(f"{len(filas)} pedidos ({args.repeticiones} repeticiones de {len(filas) // args.repeticiones} frases)")
    print(f"{'etapa':28s} {'mediana':>9s} {'p95':>9s}")
    for clave, nombre in (("ms_intent", "interpretar (/intent)"), ("ms_object", "buscar objeto (/object)"),
                          ("ms_glb", "descargar .glb"), ("ms_total", "total sin dibujar")):
        valores = [f[clave] for f in filas if f.get(clave) is not None]
        if valores:
            print(f"{nombre:28s} {statistics.median(valores):7.1f} ms {p95(valores):7.1f} ms")
    encontrados = [f for f in filas if f.get("ms_glb") is not None]
    print(f"\nModelo encontrado en {len(encontrados)} de {len(filas)} pedidos. "
          f"Detalle en {RESULTADOS / 'latencias_backend.csv'}")


if __name__ == "__main__":
    main()
