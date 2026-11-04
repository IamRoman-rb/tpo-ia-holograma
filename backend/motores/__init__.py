"""Motores generados por código, con todas sus piezas nombradas: variedades a pedido y las del catálogo.

pedido("motor V8 turbo") -> Combustion(cilindros=8, disposicion="v", turbo=True); pedido("mostrame una silla") -> None.
glb_para(parametros) escribe el .glb (si no existe) en frontend/models/motores/ y devuelve su url para el visor.
Generar un motor tarda milisegundos y siempre da el mismo resultado para los mismos parámetros.
"""
import math
import re
from dataclasses import asdict
from pathlib import Path

from geometria import a_glb
from texto import normalizar

from .combustion import Combustion
from .combustion import generar as generar_combustion
from .combustion import nombre as nombre_combustion
from .electrico import Electrico
from .electrico import generar as generar_electrico
from .electrico import nombre as nombre_electrico

RAIZ = Path(__file__).resolve().parent.parent.parent
CARPETA = RAIZ / "frontend" / "models" / "motores"
URL = "models/motores"

NUMEROS = {"uno": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
           "nueve": 9, "diez": 10, "doce": 12}
PREFIJOS = {"mono": 1, "bi": 2, "tri": 3, "tetra": 4, "cuatri": 4, "penta": 5, "hexa": 6}
ELECTRICO = re.compile(r"\b(electric\w*|trifasic\w*|induccion|corriente continua|brushless|sin escobillas)\b")
COMBUSTION = re.compile(r"\b(cilindros?|\w*cilindric[oa]|v ?-?\d{1,2}|en v|boxer|opuestos|radial|estrella|turbo\w*|"
                        r"biturbo|diesel|gasolero|gasoil|nafta\w*|naftero|combustion|explosion|de auto|de coche|"
                        r"de moto|de camion|de avion|aviacion|cuatro tiempos|valvulas|litros|cilindrada)\b")
POTENCIA = {"induccion": 4.0, "continua": 0.5, "brushless": 0.3}   # kW, si el pedido no la dice


def pedido(texto):
    """Los parámetros del motor que se pide, o None si el texto no pide un motor con alguna característica."""
    t = normalizar(texto).replace("-", " ")
    if not re.search(r"\bmotor(es)?\b", t):
        return None
    if ELECTRICO.search(t):
        tipo = "brushless" if re.search(r"brushless|sin escobillas", t) else \
            "continua" if "corriente continua" in t else "induccion"
        potencia = POTENCIA[tipo]
        if p := re.search(r"(\d+(?:[.,]\d+)?)\s*(kw|hp|cv)\b", t):
            potencia = float(p.group(1).replace(",", ".")) * (1 if p.group(2) == "kw" else 0.7355)
        return Electrico(tipo, round(min(max(potencia, 0.05), 200.0), 2), "brida" if "brida" in t else "patas")
    if not COMBUSTION.search(t):
        return None

    disposicion = ("radial" if re.search(r"radial|estrella|de avion|aviacion", t) else
                   "boxer" if re.search(r"boxer|opuestos", t) else
                   "v" if re.search(r"\bv ?\d{1,2}\b|\ben v\b", t) else "linea")
    cilindros = None
    if v := re.search(r"\bv ?(\d{1,2})\b", t):
        cilindros = int(v.group(1))
    elif n := re.search(r"\b(\d{1,2}|" + "|".join(NUMEROS) + r")\s+cilindros?\b", t):
        cilindros = int(n.group(1)) if n.group(1).isdigit() else NUMEROS[n.group(1)]
    elif p := re.search(r"\b(" + "|".join(PREFIJOS) + r")cilindric[oa]\b", t):
        cilindros = PREFIJOS[p.group(1)]
    por_defecto = {"linea": 2 if "de moto" in t else 4, "v": 8, "boxer": 4, "radial": 9}
    cilindros = cilindros or por_defecto[disposicion]
    # Lo que se puede construir: en V y bóxer, pares; radial, impares (encendido parejo); hasta 12
    if disposicion in ("v", "boxer"):
        cilindros = max(2, min(12, cilindros + cilindros % 2))
    elif disposicion == "radial":
        cilindros = max(3, min(9, cilindros + (1 - cilindros % 2)))
    else:
        cilindros = max(1, min(8, cilindros))

    diametro = carrera = 8.6
    if c := re.search(r"(\d+(?:[.,]\d+)?)\s*(litros|l|cc|cm3|cilindrada)\b", t):
        litros = float(c.group(1).replace(",", "."))
        litros = litros / 1000 if c.group(2) in ("cc", "cm3") or litros > 20 else litros
        # cilindrada = cilindros · π/4 · D² · carrera, con carrera = diámetro (motor "cuadrado"), en cm
        diametro = carrera = round((4 * litros * 1000 / (cilindros * math.pi)) ** (1 / 3), 2)
    valvulas = 4
    if v := re.search(r"\b(\d{1,2})\s+valvulas\b", t):
        total = int(v.group(1))
        valvulas = 2 if (total if total <= 4 else total / cilindros) <= 2 else 4
    return Combustion(
        cilindros=cilindros, disposicion=disposicion,
        angulo_v=60.0 if disposicion == "v" and cilindros in (6, 12) else 90.0,
        diametro=diametro, carrera=carrera,
        arboles="sohc" if re.search(r"sohc|un (solo )?arbol", t) or valvulas == 2 else "dohc",
        valvulas=valvulas, turbo=bool(re.search(r"turbo", t)),
        distribucion="cadena" if "cadena" in t else "correa",
        combustible="diesel" if re.search(r"diesel|gasolero|gasoil", t) else "nafta")


def nombre(parametros):
    return nombre_electrico(parametros) if isinstance(parametros, Electrico) else nombre_combustion(parametros)


def generar(parametros):
    return generar_electrico(parametros) if isinstance(parametros, Electrico) else generar_combustion(parametros)


def identificador(parametros):
    """Un nombre de archivo estable para cada combinación de parámetros."""
    partes = ["electrico" if isinstance(parametros, Electrico) else "combustion"]
    for valor in asdict(parametros).values():
        partes.append(str(valor).replace(".", "p") if not isinstance(valor, bool) else ("si" if valor else "no"))
    return "_".join(partes)


def glb_para(parametros, carpeta=None):
    """Escribe el .glb de estos parámetros (si no está) y devuelve (id, url relativa al frontend).
    Los del catálogo están en frontend/models/motores/ (versionados); los pedidos, en motores/a_pedido/."""
    id_ = identificador(parametros)
    ruta = (carpeta or CARPETA) / f"{id_}.glb"
    if carpeta is None and not ruta.exists():
        ruta = CARPETA / "a_pedido" / f"{id_}.glb"
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_suffix(".tmp")
        tmp.write_bytes(a_glb(generar(parametros), nombre(parametros), unidad="cm"))
        tmp.replace(ruta)
    return id_, ruta.relative_to(RAIZ / "frontend").as_posix()


# Las variedades del catálogo: aparecen en la lista del visor y funcionan sin el backend (plan B)
CATALOGO = [
    (Combustion(cilindros=1, valvulas=2, arboles="sohc"), ["motor monocilíndrico", "motor de un cilindro", "motor de moto chica"],
     "Un solo cilindro: el motor más simple, como el de una moto chica o una cortadora de césped."),
    (Combustion(cilindros=3, turbo=True), ["motor de 3 cilindros", "motor tricilíndrico turbo"],
     "Tres cilindros en línea con turbo: chico y eficiente, como el de muchos autos actuales."),
    (Combustion(cilindros=4), ["motor de auto", "motor de 4 cilindros", "motor a nafta", "motor de combustión"],
     "Cuatro cilindros en línea, doble árbol de levas y 16 válvulas: el motor de auto más común."),
    (Combustion(cilindros=4, turbo=True, combustible="diesel", distribucion="cadena"),
     ["motor diésel", "motor turbodiésel", "motor gasolero"],
     "Cuatro cilindros diésel con turbo: inyectores en lugar de bujías."),
    (Combustion(cilindros=6), ["motor de 6 cilindros en línea", "seis en línea"],
     "Seis cilindros en línea: muy equilibrado y suave."),
    (Combustion(cilindros=6, disposicion="v", angulo_v=60.0), ["motor V6", "V6"],
     "Seis cilindros en V a 60°: dos bancos de tres."),
    (Combustion(cilindros=8, disposicion="v"), ["motor V8", "V8", "motor de 8 cilindros"],
     "Ocho cilindros en V a 90°: dos bancos de cuatro, típico de autos deportivos y camionetas."),
    (Combustion(cilindros=12, disposicion="v", angulo_v=60.0, turbo=True), ["motor V12", "V12"],
     "Doce cilindros en V con dos turbos: el de los superdeportivos."),
    (Combustion(cilindros=4, disposicion="boxer"), ["motor bóxer", "motor de cilindros opuestos"],
     "Cuatro cilindros horizontales y opuestos: bajo y equilibrado."),
    (Combustion(cilindros=6, disposicion="boxer", turbo=True), ["motor bóxer de 6 cilindros"],
     "Seis cilindros opuestos con turbo."),
    (Combustion(cilindros=7, disposicion="radial", valvulas=2, arboles="sohc"), ["motor radial de 7 cilindros"],
     "Siete cilindros en estrella alrededor de un solo cigüeñal."),
    (Combustion(cilindros=9, disposicion="radial", valvulas=2, arboles="sohc", diametro=10.0, carrera=11.0),
     ["motor radial", "motor de avión", "motor en estrella"],
     "Nueve cilindros en estrella con hélice: el motor de los aviones de mediados del siglo XX."),
    (Electrico("induccion", 4.0, "brida"), ["motor trifásico con brida", "motor eléctrico con brida"],
     "Motor de inducción trifásico con brida de montaje."),
    (Electrico("continua", 0.5), ["motor de corriente continua", "motor de continua", "motor con escobillas"],
     "Motor de corriente continua: imanes, inducido, colector y escobillas."),
    (Electrico("brushless", 0.3), ["motor brushless", "motor sin escobillas", "motor de dron"],
     "Motor sin escobillas de rotor exterior, como el de los drones."),
]
# El motor eléctrico de siempre (id motor_electrico) ahora es el trifásico de inducción con patas
TRIFASICO = Electrico("induccion", 4.0, "patas")
