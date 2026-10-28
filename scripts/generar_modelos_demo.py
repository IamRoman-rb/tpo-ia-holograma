"""Genera el catálogo precargado de modelos .glb (plan B de la demo).

Construye piezas industriales a partir de primitivas (cilindros, toroides, esferas, prismas)
con la geometría de backend/geometria.py. Cada modelo está formado por piezas con nombre,
descripción, orden y dirección de desmontaje: el visor puede desarmarlo pieza por pieza.
También genera frontend/models/catalogo.json.

Uso (desde la raíz):  python scripts/generar_modelos_demo.py
"""
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "backend"))

from geometria import (Modelo, caja, cilindro, circulo, esfera, escribir_glb, mover,  # noqa: E402
                       perfil_engranaje, prisma_anular, rot_x, rot_z, toroide)

SALIDA = RAIZ / "frontend" / "models"

ACERO = ((0.80, 0.80, 0.82), 1.0, 0.25)
HIERRO = ((0.25, 0.27, 0.30), 0.7, 0.45)
CROMO = ((0.92, 0.92, 0.94), 1.0, 0.08)
HORIZ_X = rot_z(math.pi / 2)  # lleva un cilindro del eje Y al eje X
FRENTE = rot_x(math.pi / 2)   # lleva un cilindro del eje Y al eje Z (de frente a la cámara)

ARRIBA, ABAJO = (0, 1, 0), (0, -1, 0)
IZQUIERDA, DERECHA = (-1, 0, 0), (1, 0, 0)
ADELANTE, ATRAS = (0, 0, 1), (0, 0, -1)


def valvula():
    m = Modelo({"hierro": HIERRO, "volante": ((0.75, 0.08, 0.06), 0.2, 0.4), "acero": ACERO,
                "bulones": ((0.55, 0.55, 0.55), 0.9, 0.35)})
    m.pieza("volante", "Volante", "Rueda de accionamiento: al girarla sube o baja la compuerta.", 1, ARRIBA)
    m.agregar("volante", mover(toroide(0.38, 0.035), (0, 1.28, 0)))
    m.agregar("volante", mover(cilindro(0.08, 0.08, 24), (0, 1.28, 0)))
    m.agregar("volante", mover(caja(0.76, 0.04, 0.04), (0, 1.28, 0)))
    m.agregar("volante", mover(caja(0.04, 0.04, 0.76), (0, 1.28, 0)))
    m.pieza("vastago", "Vástago", "Eje roscado que transmite el giro del volante a la compuerta.", 2, ARRIBA)
    m.agregar("acero", mover(cilindro(0.05, 0.6, 16), (0, 1.0, 0)))
    m.pieza("bonete", "Bonete", "Tapa superior del cuerpo; aloja el vástago y su empaquetadura.", 3, ARRIBA)
    m.agregar("hierro", mover(cilindro(0.32, 0.08), (0, 0.32, 0)))
    m.agregar("hierro", mover(cilindro(0.22, 0.5), (0, 0.55, 0)))
    for orden, (x, lado, direccion) in enumerate(((-0.6, "izquierda", IZQUIERDA), (0.6, "derecha", DERECHA))):
        m.pieza(f"bulones_{lado}", f"Bulones de la brida {lado}",
                "Ocho bulones que fijan la brida a la cañería.", 4 + orden, direccion)
        for k in range(8):
            a = 2 * math.pi * k / 8
            m.agregar("bulones", mover(cilindro(0.04, 0.16, 12), (x, 0.45 * math.cos(a), 0.45 * math.sin(a)),
                                       HORIZ_X))
    for orden, (x, lado, direccion) in enumerate(((-0.6, "izquierda", IZQUIERDA), (0.6, "derecha", DERECHA))):
        m.pieza(f"brida_{lado}", f"Brida {lado}", "Disco de unión con la cañería.", 6 + orden, direccion)
        m.agregar("hierro", mover(cilindro(0.55, 0.1), (x, 0, 0), HORIZ_X))
    m.pieza("cuerpo", "Cuerpo", "Carcasa de hierro fundido por donde pasa el fluido. Queda fija.")
    m.agregar("hierro", mover(cilindro(0.32, 1.2), R=HORIZ_X))
    m.agregar("hierro", esfera(0.42))
    return m


def engranaje():
    m = Modelo({"acero": ACERO})
    m.pieza("cubo", "Cubo", "Manguito central que se monta sobre el eje.", 1, ADELANTE)
    m.agregar("acero", mover(prisma_anular(circulo(0.35), 0.2, 0.45), R=FRENTE))
    m.pieza("corona", "Corona dentada", "Disco con 24 dientes rectos que transmite el giro. Queda fija.")
    m.agregar("acero", mover(prisma_anular(perfil_engranaje(24, 0.8, 0.95), 0.36, 0.25), R=FRENTE))
    return m


def tornillo():
    m = Modelo({"zinc": ((0.72, 0.74, 0.76), 0.9, 0.3), "laton": ((0.78, 0.60, 0.25), 0.9, 0.3)})
    m.pieza("tuerca", "Tuerca", "Tuerca hexagonal que ajusta la unión.", 1, ABAJO)
    m.agregar("laton", mover(cilindro(0.3, 0.22, 6, plano=True), (0, -0.35, 0)))
    m.pieza("tornillo", "Tornillo", "Cabeza hexagonal y vástago roscado.", 2, ARRIBA)
    m.agregar("zinc", mover(cilindro(0.32, 0.22, 6, plano=True), (0, 0.9, 0)))
    m.agregar("zinc", cilindro(0.15, 1.5, 32))
    for k in range(21):
        m.agregar("zinc", mover(toroide(0.15, 0.025, 32, 8), (0, -0.7 + 0.06 * k, 0)))
    m.pieza("arandela", "Arandela", "Reparte la presión de la cabeza sobre la superficie. Queda fija.")
    m.agregar("zinc", mover(prisma_anular(circulo(0.4), 0.17, 0.04), (0, 0.77, 0)))
    return m


def rodamiento():
    m = Modelo({"acero": ACERO, "bolas": CROMO})
    m.pieza("aro_exterior", "Aro exterior", "Pista exterior: se monta en el alojamiento.", 1, ATRAS)
    m.agregar("acero", mover(prisma_anular(circulo(1.0, 96), 0.82, 0.35), R=FRENTE))
    m.pieza("bolas", "Bolas", "Doce bolas de acero cromado que ruedan entre las pistas.", 2, ADELANTE)
    for k in range(12):
        a = 2 * math.pi * k / 12
        m.agregar("bolas", mover(esfera(0.15), (0.66 * math.cos(a), 0.66 * math.sin(a), 0)))
    m.pieza("aro_interior", "Aro interior", "Pista interior: se monta sobre el eje. Queda fija.")
    m.agregar("acero", mover(prisma_anular(circulo(0.5, 96), 0.3, 0.35), R=FRENTE))
    return m


def motor():
    m = Modelo({"carcasa": ((0.10, 0.30, 0.60), 0.3, 0.5), "acero": ACERO, "base": HIERRO,
                "cobre": ((0.72, 0.42, 0.20), 0.9, 0.35)})
    m.pieza("caja_bornes", "Caja de bornes", "Caja de conexiones eléctricas del motor.", 1, ARRIBA)
    m.agregar("carcasa", mover(caja(0.35, 0.25, 0.35), (0, 0.62, 0)))
    m.pieza("tapa_trasera", "Tapa trasera", "Escudo del lado del ventilador; sostiene un rodamiento.", 2, IZQUIERDA)
    m.agregar("carcasa", mover(cilindro(0.5, 0.15), (-0.72, 0, 0), HORIZ_X))
    m.pieza("tapa_delantera", "Tapa delantera", "Escudo del lado del eje; sostiene el otro rodamiento.", 3, DERECHA)
    m.agregar("carcasa", mover(cilindro(0.5, 0.15), (0.72, 0, 0), HORIZ_X))
    m.pieza("rotor", "Rotor y eje", "Parte que gira: el bobinado del rotor y el eje de salida.", 4, DERECHA)
    m.agregar("cobre", mover(cilindro(0.38, 1.1), R=HORIZ_X))
    m.agregar("acero", mover(cilindro(0.07, 1.7, 24), (0.25, 0, 0), HORIZ_X))
    m.pieza("base", "Base de montaje", "Patas para fijar el motor.", 5, ABAJO)
    m.agregar("base", mover(caja(1.0, 0.1, 0.9), (0, -0.75, 0)))
    for x in (-0.35, 0.35):
        m.agregar("base", mover(caja(0.15, 0.25, 0.7), (x, -0.6, 0)))
    m.pieza("carcasa", "Carcasa", "Estator con aletas de refrigeración. Queda fija.")
    m.agregar("carcasa", mover(prisma_anular(circulo(0.55), 0.4, 1.3), R=HORIZ_X))
    for k in range(16):
        a = 2 * math.pi * k / 16
        m.agregar("carcasa", mover(caja(1.1, 0.12, 0.04), (0, 0.6 * math.cos(a), 0.6 * math.sin(a)), rot_x(a)))
    return m


CATALOGO = [
    ("valvula_compuerta", valvula, "Válvula de compuerta",
     ["válvula", "valvula de acero", "llave de paso", "válvula industrial", "válvula con volante"],
     "Válvula de compuerta bridada con volante de accionamiento."),
    ("engranaje_recto", engranaje, "Engranaje recto",
     ["engranaje", "rueda dentada", "piñón", "engrane", "corona dentada"],
     "Engranaje recto de 24 dientes con cubo central."),
    ("tornillo_hexagonal", tornillo, "Tornillo hexagonal con tuerca",
     ["tornillo", "bulón", "perno", "tuerca", "tornillo con tuerca"],
     "Tornillo de cabeza hexagonal con arandela y tuerca."),
    ("rodamiento_bolas", rodamiento, "Rodamiento de bolas",
     ["rodamiento", "ruleman", "rulemán", "cojinete", "balero"],
     "Rodamiento rígido de una hilera de bolas."),
    ("motor_electrico", motor, "Motor eléctrico",
     ["motor", "motor trifásico", "motor industrial", "motor con aletas", "motor eléctrico industrial"],
     "Motor eléctrico con carcasa aletada, eje y base de montaje."),
]


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    indice = []
    for id_, fabricar, nombre, sinonimos, descripcion in CATALOGO:
        ruta = SALIDA / f"{id_}.glb"
        modelo = fabricar()
        escribir_glb(modelo, ruta, id_)
        indice.append({"id": id_, "nombre": nombre, "sinonimos": sinonimos, "descripcion": descripcion,
                       "url": f"models/{id_}.glb", "origen": "catalogo"})
        print(f"{ruta.name:28s} {ruta.stat().st_size / 1024:7.1f} KB · {len(modelo.piezas)} piezas")
    # Conservar los modelos generados por IA que ya estaban en el catálogo
    ruta_indice = SALIDA / "catalogo.json"
    if ruta_indice.exists():
        anteriores = json.loads(ruta_indice.read_text(encoding="utf-8"))
        indice += [it for it in anteriores if it.get("origen") == "generado"]
    with open(ruta_indice, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"catalogo.json con {len(indice)} modelos")


if __name__ == "__main__":
    main()
