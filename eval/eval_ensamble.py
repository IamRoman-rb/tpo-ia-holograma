"""Evalúa la generación 3D por piezas (backend/ensamble.py) con pedidos de pedidos_ensamble.csv.

Mide por pedido: si se generó, segundos, tokens y costo, piezas (desmontables y fijas), formas
descartadas por la validación, partes clave presentes, si el tamaño es razonable y el peso del .glb.
Los cinco primeros pedidos son los mismos que se evalúan con TripoSR y Shap-E en Colab.

La calidad la puntúan personas, de 1 a 5, en las columnas vacías de eval/resultados/ensamble.csv
(rúbrica en eval/generacion3d/README.md). Cada modelo se abre en el visor con la URL de la columna "ver".

Uso (desde la raíz, con el entorno del backend):
    python eval/eval_ensamble.py                         # los 10 pedidos: 10 llamadas a la API (tiene costo)
    python eval/eval_ensamble.py --solo bomba_agua,morsa
    python eval/eval_ensamble.py --prueba                # sin API: responde siempre la bomba de ejemplo
"""
import argparse
import csv
import json
import re
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(RAIZ / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

import anthropic  # noqa: E402

import ensamble  # noqa: E402
import geometria as g  # noqa: E402
from texto import normalizar  # noqa: E402

PEDIDOS = Path(__file__).parent / "pedidos_ensamble.csv"
RESULTADOS = Path(__file__).parent / "resultados" / "ensamble.csv"
MODELOS = RAIZ / "frontend" / "models" / "evaluacion"   # el visor los abre con ?modelo=
EJEMPLO = RAIZ / "backend" / "tests" / "datos" / "ensamble_bomba.json"

# USD por millón de tokens (entrada, salida), precios de octubre de 2026.
# Si el pedido cae en un modelo de respaldo, se cobra a la tarifa de ese modelo.
PRECIOS = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
PUNTAJES = ["calidad_forma", "calidad_detalle", "desarmado_logico"]  # los completan personas


class ClienteMedido:
    """Envuelve el cliente del SDK y guarda la última respuesta, para leer el uso de tokens."""

    def __init__(self, cliente):
        self.ultima = None

        def create(**kwargs):
            self.ultima = cliente.beta.messages.create(**kwargs)
            return self.ultima

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def cliente_de_prueba():
    texto = EJEMPLO.read_text(encoding="utf-8")
    respuesta = SimpleNamespace(stop_reason="end_turn", model="prueba",
                                content=[SimpleNamespace(type="text", text=texto)],
                                usage=SimpleNamespace(input_tokens=0, output_tokens=0, iterations=None))
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **_: respuesta)))


def consumo(respuesta):
    """Tokens de entrada y salida y costo en USD (None si el modelo no está en PRECIOS)."""
    uso = respuesta.usage
    # Con respaldo del lado del servidor, usage.iterations trae un intento por modelo
    intentos = getattr(uso, "iterations", None) or [uso]
    entrada = salida = 0
    costo = 0.0
    for intento in intentos:
        modelo = getattr(intento, "model", None) or respuesta.model
        e, s = getattr(intento, "input_tokens", 0) or 0, getattr(intento, "output_tokens", 0) or 0
        entrada, salida = entrada + e, salida + s
        precio = PRECIOS.get(modelo)
        costo = None if costo is None or precio is None else costo + (e * precio[0] + s * precio[1]) / 1e6
    return entrada, salida, costo


def medida_mayor(glb: bytes) -> float:
    """Lado mayor de la caja que contiene al modelo, en cm (la unidad del ensamble)."""
    largo = struct.unpack_from("<I", glb, 12)[0]
    gltf = json.loads(glb[20:20 + largo])
    minimos = [a["min"] for a in gltf["accessors"] if a.get("type") == "VEC3" and "min" in a]
    maximos = [a["max"] for a in gltf["accessors"] if a.get("type") == "VEC3" and "max" in a]
    return max(max(m[i] for m in maximos) - min(m[i] for m in minimos) for i in range(3))


def partes_presentes(diseno, partes_clave: str):
    """Cuántas partes clave aparecen en el nombre o el id de alguna pieza. Cada parte acepta
    sinónimos separados por "|" ("impulsor|rodete")."""
    textos = [normalizar(f"{p.nombre} {p.id.replace('_', ' ')}") for p in diseno.piezas]
    encontradas, faltan = 0, []
    for parte in partes_clave.split(";"):
        opciones = [normalizar(o) for o in parte.split("|")]
        if any(re.search(rf"\b{re.escape(o)}", t) for o in opciones for t in textos):
            encontradas += 1
        else:
            faltan.append(parte.split("|")[0])
    return encontradas, len(partes_clave.split(";")), faltan


def evaluar(fila, cliente, carpeta):
    medido = ClienteMedido(cliente)
    salida = {"id": fila["id"], "pedido": fila["pedido"], "en_colab": fila["en_colab"]}
    t0 = time.perf_counter()
    try:
        crudo = ensamble.pedir_ensamble(fila["pedido"], medido)
    except (ensamble.EnsambleInvalido, anthropic.AnthropicError) as e:
        salida.update(ok=False, error=str(e) or type(e).__name__, segundos=round(time.perf_counter() - t0, 1))
        return salida
    segundos = time.perf_counter() - t0
    entrada, tokens_salida, costo = consumo(medido.ultima)
    salida.update(segundos=round(segundos, 1), modelo=medido.ultima.model,
                  tokens_entrada=entrada, tokens_salida=tokens_salida,
                  costo_usd=None if costo is None else round(costo, 4))
    try:
        diseno = ensamble.validar(crudo)
    except ensamble.EnsambleInvalido as e:
        salida.update(ok=False, error=f"validación: {e}")
        return salida

    glb = g.a_glb(ensamble.a_modelo(diseno), diseno.nombre)
    (carpeta / f"{fila['id']}.glb").write_bytes(glb)
    recibidas = sum(len(p.get("formas", [])) for p in crudo.get("piezas", [])[:ensamble.MAX_PIEZAS])
    formas = sum(len(p.formas) for p in diseno.piezas)
    desmontables = sum(p.orden_desmontaje is not None for p in diseno.piezas)
    encontradas, total, faltan = partes_presentes(diseno, fila["partes_clave"])
    mayor = medida_mayor(glb)
    salida.update(
        ok=True, error="", nombre=diseno.nombre, piezas=len(diseno.piezas), desmontables=desmontables,
        fijas=len(diseno.piezas) - desmontables, formas=formas, formas_descartadas=recibidas - formas,
        partes_clave=f"{encontradas}/{total}", faltan="; ".join(faltan),
        medida_mayor_cm=round(mayor, 1),
        escala_ok=float(fila["medida_min_cm"]) <= mayor <= float(fila["medida_max_cm"]),
        kb=round(len(glb) / 1024, 1),
        ver=f"http://localhost:5500/?modelo=models/evaluacion/{fila['id']}.glb",
    )
    return salida


def guardar(filas_nuevas, orden):
    """Reemplaza en el CSV solo los pedidos que se corrieron; conserva el resto."""
    anteriores = {}
    if RESULTADOS.exists():
        with open(RESULTADOS, encoding="utf-8") as f:
            anteriores = {r["id"]: r for r in csv.DictReader(f)}
    for fila in filas_nuevas:
        anteriores[fila["id"]] = {**fila, **{p: "" for p in PUNTAJES}}  # modelo nuevo: puntaje nuevo
    columnas = list(dict.fromkeys(c for r in anteriores.values() for c in r if c not in PUNTAJES)) + PUNTAJES
    RESULTADOS.parent.mkdir(exist_ok=True)
    with open(RESULTADOS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(anteriores[i] for i in orden if i in anteriores)


def resumen(filas):
    buenas = [f for f in filas if f["ok"]]
    print(f"\nGenerados: {len(buenas)}/{len(filas)}")
    for f in filas:
        if not f["ok"]:
            print(f"  FALLA  {f['id']}: {f['error']}")
    if not buenas:
        return
    encontradas = sum(int(f["partes_clave"].split("/")[0]) for f in buenas)
    total = sum(int(f["partes_clave"].split("/")[1]) for f in buenas)
    print(f"Segundos por modelo: mediana {statistics.median(f['segundos'] for f in buenas):.1f}"
          f" · máximo {max(f['segundos'] for f in buenas):.1f}")
    print(f"Piezas por modelo: mediana {statistics.median(f['piezas'] for f in buenas):g}"
          f" · desmontables {sum(f['desmontables'] for f in buenas)}/{sum(f['piezas'] for f in buenas)}")
    print(f"Partes clave presentes: {encontradas}/{total} ({100 * encontradas / total:.0f} %)")
    print(f"Tamaño razonable: {sum(f['escala_ok'] for f in buenas)}/{len(buenas)}")
    print(f"Formas descartadas por la validación: {sum(f['formas_descartadas'] for f in buenas)}"
          f" de {sum(f['formas'] + f['formas_descartadas'] for f in buenas)}")
    costos = [f["costo_usd"] for f in filas if f.get("costo_usd") is not None]
    if costos:
        print(f"Costo: {sum(costos):.3f} USD en total · {statistics.mean(costos):.3f} USD por pedido")
    for f in buenas:
        faltan = f" · faltan: {f['faltan']}" if f["faltan"] else ""
        print(f"  {f['id']:<20} {f['piezas']:>2} piezas · {f['segundos']:>5.1f} s"
              f" · {f['medida_mayor_cm']:>6.1f} cm{'' if f['escala_ok'] else ' (fuera de rango)'}{faltan}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", help="ids separados por coma (ver eval/pedidos_ensamble.csv)")
    ap.add_argument("--prueba", action="store_true",
                    help="sin API ni costo: prueba el circuito con la bomba de ejemplo y no guarda resultados")
    args = ap.parse_args()

    with open(PEDIDOS, encoding="utf-8") as f:
        pedidos = list(csv.DictReader(f))
    orden = [p["id"] for p in pedidos]
    if args.solo:
        elegidos = set(args.solo.split(","))
        if desconocidos := elegidos - set(orden):
            sys.exit(f"ids desconocidos: {', '.join(sorted(desconocidos))}")
        pedidos = [p for p in pedidos if p["id"] in elegidos]

    if args.prueba:
        cliente = cliente_de_prueba()
        carpeta = Path(tempfile.mkdtemp(prefix="eval_ensamble_"))
    else:
        cliente = anthropic.Anthropic(timeout=180.0, max_retries=1)
        try:
            cliente.models.list(limit=1)
        except (anthropic.AnthropicError, TypeError) as e:  # TypeError: el SDK no encontró credenciales
            sys.exit(f"LLM no disponible ({type(e).__name__}). Configurá ANTHROPIC_API_KEY en .env.")
        carpeta = MODELOS
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"Modelo: {ensamble.MODELO} · {len(pedidos)} pedidos (cada uno es una llamada a la API)")

    filas = []
    for fila in pedidos:
        print(f"→ {fila['pedido']}…", flush=True)
        filas.append(evaluar(fila, cliente, carpeta))
    resumen(filas)

    if args.prueba:
        print(f"\n(prueba: no se guardaron resultados; los .glb quedaron en {carpeta})")
    else:
        guardar(filas, orden)
        print(f"\nResultados: {RESULTADOS.relative_to(RAIZ)} · modelos: {carpeta.relative_to(RAIZ)}")
        print("Para puntuar: levantar el frontend y abrir la URL de la columna \"ver\" de cada fila.")


if __name__ == "__main__":
    main()
