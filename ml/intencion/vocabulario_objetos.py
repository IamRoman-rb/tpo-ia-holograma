"""Arma el vocabulario de objetos para entrenar el etiquetador del intérprete: miles de nombres en español.

Fuente: WordNet en español del Open Multilingual Wordnet (Multilingual Central Repository, CC BY 3.0), vía nltk.
Se toman los sustantivos que cuelgan de categorías de cosas que se pueden mostrar en 3D: artefactos (aparatos,
máquinas, herramientas, recipientes, muebles, vehículos), construcciones, ropa, componentes, y también animales,
partes del cuerpo, frutas y verduras. Se descartan las palabras que quedaron sin traducir (iguales al inglés),
las raras (con mayúsculas, guiones o más de cuatro palabras) y las que nombran los objetos de los sets de
evaluación: el modelo se mide con objetos que nunca vio.

El género y el número se deducen de la terminación (sirve para elegir "un/una/unos"; algún error deja frases
como las que diría alguien que se equivoca, no cambia qué palabras son el objeto).

Uso (desde la raíz, una vez):  backend/.venv/Scripts/python -m pip install nltk
                               backend/.venv/Scripts/python ml/intencion/vocabulario_objetos.py
Salida: ml/intencion/objetos.csv (se versiona, así generar_frases.py no necesita nltk)
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SALIDA = AQUI / "objetos.csv"

# (synset raíz, categoría): se toman todos sus hipónimos
CATEGORIAS = [
    ("instrumentality.n.03", "artefacto"), ("structure.n.01", "construccion"), ("covering.n.02", "cobertura"),
    ("commodity.n.01", "producto"), ("component.n.03", "componente"), ("animal.n.01", "animal"),
    ("body_part.n.01", "cuerpo"), ("edible_fruit.n.01", "fruta"), ("vegetable.n.01", "verdura"),
]
SETS_EVAL = ["frases_prueba.csv", "frases_validacion.csv", "frases_nuevas.csv", "frases_objetos_validacion.csv",
            "frases_objetos_prueba.csv", "frases_comandos_validacion.csv", "frases_comandos_prueba.csv"]

# Sustantivos en -a que son masculinos y en -o que son femeninos (los comunes)
MASCULINOS_EN_A = {"dia", "mapa", "sistema", "problema", "programa", "tema", "planeta", "cometa", "clima",
                   "esquema", "diagrama", "telegrama", "pijama", "sofa", "tranvia", "prisma", "diafragma",
                   "fantasma", "aroma", "poema", "emblema", "radar", "panda", "puma", "koala", "gorila",
                   "yoga", "drama", "idioma", "carisma", "estigma", "dogma", "enigma", "lema", "cisma"}
FEMENINOS_EN_O = {"mano", "radio", "foto", "moto", "nao", "seo", "libido"}


def sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def genero(nombre):
    """m, f, mp o fp según la primera palabra (el núcleo del nombre)."""
    nucleo = sin_tildes(nombre.split()[0])
    plural = (len(nucleo) > 4 and nucleo.endswith(("as", "os", "es")) and not nucleo.endswith(("ies", "is"))
              and not re.search(r"[áéíóú]s$", nombre.split()[0]))
    singular = nucleo[:-1] if plural else nucleo
    if singular.endswith("e") and plural and singular[:-1].endswith(("on", "dad", "or", "al", "il")):
        singular = singular[:-1]     # "camiones" -> "camion", "tenedores" -> "tenedor"
    if singular in MASCULINOS_EN_A:
        g = "m"
    elif singular in FEMENINOS_EN_O:
        g = "f"
    elif singular.endswith(("a", "ion", "dad", "tad", "tud", "umbre", "sis", "ie")):
        g = "f"
    else:
        g = "m"
    return g + ("p" if plural else "")


def nucleos_de_evaluacion():
    """Primera palabra de cada objeto de los sets de evaluación (y de las palabras clave de la prueba)."""
    nucleos = set()
    for nombre in SETS_EVAL:
        with open(RAIZ / "eval" / nombre, encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                for columna in ("objeto", "objeto_clave"):
                    if fila.get(columna):
                        palabras = re.findall(r"[a-z0-9]+", sin_tildes(fila[columna]))
                        if palabras:
                            nucleos.add(palabras[0])
    return nucleos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from nltk.corpus import wordnet as wn

    prohibidos = nucleos_de_evaluacion()
    vistos, filas = set(), []
    for raiz, categoria in CATEGORIAS:
        synsets = set(wn.synset(raiz).closure(lambda s: s.hyponyms())) | {wn.synset(raiz)}
        for s in sorted(synsets, key=lambda s: s.name()):
            ingles = {sin_tildes(l.replace("_", " ")) for l in s.lemma_names("eng")}
            for lema in s.lemma_names("spa"):
                nombre = lema.replace("_", " ").strip()
                clave = sin_tildes(nombre)
                palabras = clave.split()
                if (clave in vistos or nombre != nombre.lower() or not re.fullmatch(r"[a-záéíóúüñ ]+", nombre)
                        or not 1 <= len(palabras) <= 4 or len(clave) < 3 or clave in ingles
                        or palabras[0] in prohibidos or palabras[0] in {"de", "del", "la", "el", "los", "las"}):
                    continue
                vistos.add(clave)
                filas.append({"nombre": nombre, "genero": genero(nombre), "categoria": categoria})

    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nombre", "genero", "categoria"])
        w.writeheader()
        w.writerows(filas)
    por_categoria = {}
    for fila in filas:
        por_categoria[fila["categoria"]] = por_categoria.get(fila["categoria"], 0) + 1
    print(f"{len(filas)} objetos -> {SALIDA.relative_to(RAIZ)}  {por_categoria}")
    print(f"excluidos por estar en la evaluación: {sorted(prohibidos)}")


if __name__ == "__main__":
    main()
