"""Resume las planillas de pruebas con usuarios para docs/metricas.md (sección 15).

Uso (desde la raíz):  python eval/pruebas_usuarios/resumir.py [directorio_de_planillas]
Por defecto lee eval/resultados/pruebas_usuarios/.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DIRECTORIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "resultados" / "pruebas_usuarios"


def leer(nombre):
    ruta = DIRECTORIO / nombre
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8") as f:
        return [fila for fila in csv.DictReader(f) if any(v.strip() for k, v in fila.items() if k != "participante")]


tareas = [f for f in leer("planilla_tareas.csv") if f["completo"].strip()]
if tareas:
    print("== Tareas (meta: ≥ 80 % sin ayuda)")
    por_tarea = defaultdict(list)
    for f in tareas:
        por_tarea[f["tarea"]].append(f)
    for tarea, filas in sorted(por_tarea.items()):
        sin_ayuda = sum(f["completo"].strip() == "si" for f in filas)
        segundos = [float(f["segundos"]) for f in filas if f["segundos"].strip()]
        mediana = sorted(segundos)[len(segundos) // 2] if segundos else float("nan")
        print(f"  {tarea}: {sin_ayuda}/{len(filas)} sin ayuda · mediana {mediana:.0f} s")
    total = sum(f["completo"].strip() == "si" for f in tareas)
    print(f"  TOTAL: {total}/{len(tareas)} ({100 * total / len(tareas):.0f} %) · {len({f['participante'] for f in tareas})} personas")

gestos = [f for f in leer("planilla_gestos.csv") if f["reconocidos"].strip()]
if gestos:
    print("\n== Gestos (meta: ≥ 90 %)")
    por_gesto = defaultdict(lambda: [0, 0])
    por_condicion = defaultdict(lambda: [0, 0])
    for f in gestos:
        n, ok = int(f["intentos"]), int(f["reconocidos"])
        por_gesto[f["gesto"]][0] += ok
        por_gesto[f["gesto"]][1] += n
        por_condicion[(f["luz"], f["mano"])][0] += ok
        por_condicion[(f["luz"], f["mano"])][1] += n
    for gesto, (ok, n) in por_gesto.items():
        print(f"  {gesto:18s} {ok}/{n} ({100 * ok / n:.0f} %)")
    print("  Por condición (tabla de sesgos de docs/uso_responsable.md):")
    for (luz, mano), (ok, n) in sorted(por_condicion.items()):
        print(f"    luz {luz:6s} · mano {mano:9s} {100 * ok / n:.0f} % ({n} intentos)")

falsos = [int(f["falsos_positivos_60s"]) for f in leer("planilla_gestos.csv") if f["falsos_positivos_60s"].strip()]
if falsos:
    print(f"\n== Falsos positivos en 60 s en reposo (meta: ≤ 2): máximo {max(falsos)}, promedio {sum(falsos) / len(falsos):.1f}")

if not (tareas or gestos or falsos):
    print(f"No hay planillas completas en {DIRECTORIO}")
