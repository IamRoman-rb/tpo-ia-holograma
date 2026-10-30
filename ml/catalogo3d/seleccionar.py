"""Elige qué modelos de Objaverse entran al catálogo 3D, con nombres en español, hasta un tope de GB.

Objaverse (Deitke et al., Allen AI, ODC-By 1.0): ~800 mil modelos .glb de Sketchfab, cada uno con su licencia.
Se aceptan CC BY, CC BY-SA y CC0 (se puede mostrar con atribución); nada con restricción de edad.

Dos fuentes, por rondas, hasta llenar el tope:
  - dominio (ml/catalogo3d/dominio.py): máquinas y piezas, buscadas por palabras en el nombre y los tags;
  - las 1.156 categorías LVIS de Objaverse (46 mil modelos anotados a mano), para cubrir lo que se pida.
En cada ronda entra el siguiente mejor modelo de cada categoría (calidad = destacado por Sketchfab, likes, vistas y
detalle), así el tope reparte modelos entre todas en vez de llenarse con unas pocas.

Nombres en español: los del dominio están escritos a mano; los de LVIS salen del WordNet en español (Open
Multilingual Wordnet, el mismo de ml/intencion/vocabulario_objetos.py), con algunos corregidos a mano. La descripción
en inglés es la de Cap3D (Luo et al., ODC-By 1.0) y sirve para los embeddings multilingües.

Para la evaluación (eval/eval_catalogo3d.py) se reserva un sinónimo de ~15 % de las categorías LVIS: no se usa para
nombrar ningún modelo y después se pide con él.

Uso (desde la raíz):  backend/.venv/Scripts/python ml/catalogo3d/seleccionar.py [--gb 50]
Salida: ml/catalogo3d/seleccion.csv (versionado). Los .glb se bajan con scripts/descargar_modelos3d.py.
"""
import argparse
import csv
import glob
import gzip
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import HF, METADATOS, RAIZ, SELECCION, descargar  # noqa: E402
from dominio import DOMINIO  # noqa: E402
from wordfreq import zipf_frequency  # noqa: E402

LICENCIAS = {"by": "CC BY 4.0", "by-sa": "CC BY-SA 4.0", "cc0": "CC0 1.0"}
MAX_MB = 30
CARAS = (3_000, 1_500_000)    # con detalle, pero que una gráfica integrada lo pueda mostrar
POR_CATEGORIA = {"dominio": 40, "lvis": 8}
RESERVA_EVAL = 0.15
FRECUENCIA_EVAL = 2.5         # Zipf en español (wordfreq): 2,5 es una palabra rara pero conocida
# WordNet no siempre da el nombre que se usa en Argentina, o elige otro sentido de la palabra
NOMBRES_LVIS = {
    "car_(automobile)": ["auto", "coche", "automóvil"], "refrigerator": ["heladera", "refrigerador"],
    "washer_(automatic_washer)": ["lavarropas"], "bus_(vehicle)": ["colectivo", "ómnibus"],
    "cellular_telephone": ["celular", "teléfono celular"], "computer_keyboard": ["teclado"],
    "bat_(animal)": ["murciélago"], "baseball_bat": ["bate de béisbol", "bate"], "bow_(weapon)": ["arco"],
    "crane": ["grúa"], "pickup_truck": ["camioneta"], "truck": ["camión"], "motorcycle": ["moto", "motocicleta"],
    "bicycle": ["bicicleta"], "airplane": ["avión"], "television_set": ["televisor", "tele"],
    "laptop_computer": ["notebook", "computadora portátil"], "sofa": ["sillón", "sofá"],
    "faucet": ["canilla", "grifo"], "mug": ["taza"], "trousers": ["pantalón"], "sweater": ["buzo", "suéter"],
    "seahorse": ["caballito de mar", "hipocampo"], "cash_register": ["caja registradora", "registradora"],
    "steering_wheel": ["volante", "manubrio"], "earring": ["aro", "arito", "arete"],
    "Christmas_tree": ["árbol de navidad", "arbolito"], "Lego": ["lego", "bloques de lego"],
    "bath_mat": ["alfombra de baño"], "belt_buckle": ["hebilla"], "birdbath": ["bebedero de pájaros"],
    "birdfeeder": ["comedero de pájaros"], "birthday_card": ["tarjeta de cumpleaños"], "bolo_tie": ["corbata de lazo"],
    "boom_microphone": ["micrófono de jirafa", "micrófono"], "bottle_cap": ["tapita", "tapa de botella"],
    "bowling_ball": ["bola de bowling"], "brass_plaque": ["placa de bronce"], "carton": ["cartón", "caja de cartón"],
    "clothes_hamper": ["cesto de ropa"], "coaster": ["posavasos"], "corkboard": ["corcho", "tablero de corcho"],
    "coverall": ["mameluco", "overol"], "crate": ["cajón", "jaula"], "dirt_bike": ["moto de cross", "moto enduro"],
    "dove": ["paloma"], "duffel_bag": ["bolso de viaje"], "dumpster": ["volquete", "contenedor de basura"],
    "fighter_jet": ["avión de caza", "caza"], "fishbowl": ["pecera"], "football_(American)": ["pelota de fútbol americano"],
    "football_helmet": ["casco de fútbol americano"], "freshener": ["aromatizante"], "fume_hood": ["campana de laboratorio"],
    "garden_hose": ["manguera"], "grits": ["sémola"], "halter_top": ["top"], "hand_towel": ["toalla de mano"],
    "hardback_book": ["libro de tapa dura", "libro"], "horned_cow": ["vaca con cuernos", "vaca"],
    "horse_buggy": ["sulky", "carro tirado por caballos"], "horse_carriage": ["carruaje"], "jersey": ["camiseta"],
    "knocker_(on_a_door)": ["llamador de puerta", "aldaba"], "lab_coat": ["guardapolvo", "bata de laboratorio"],
    "mail_slot": ["ranura del correo"], "mashed_potato": ["puré de papas", "puré"], "nightshirt": ["camisón"],
    "ottoman": ["puf", "otomana"], "pinwheel": ["molinete", "rehilete"], "pipe_bowl": ["pipa"],
    "pirate_flag": ["bandera pirata"], "potholder": ["agarradera"], "racket": ["raqueta"],
    "river_boat": ["barco de río", "lancha"], "runner_(carpet)": ["alfombra de pasillo"],
    "saddle_blanket": ["mandil de montura"], "shaker": ["coctelera", "salero"], "shot_glass": ["vaso de shot"],
    "shoulder_bag": ["bandolera", "morral"], "shower_head": ["flor de ducha", "regadera"], "ski_parka": ["campera de ski"],
    "ski_pole": ["bastón de ski"], "sportswear": ["ropa deportiva"], "step_stool": ["banquito", "escalerita"],
    "stirrer": ["revolvedor"], "stop_sign": ["cartel de pare", "señal de pare"], "street_sign": ["cartel de calle"],
    "string_cheese": ["queso en hebras"], "sunhat": ["sombrero de sol", "capelina"], "table-tennis_table": ["mesa de ping pong"],
    "tank_top_(clothing)": ["musculosa"], "toaster_oven": ["horno eléctrico"], "trophy_cup": ["trofeo", "copa"],
    "walking_cane": ["bastón"], "water_cooler": ["dispenser de agua", "bebedero"], "wet_suit": ["traje de neopreno"],
    "wind_chime": ["campanita de viento", "llamador de ángeles"], "window_box_(for_plants)": ["jardinera"],
    "wine_bucket": ["frapera", "cubo de vino"], "wooden_spoon": ["cuchara de madera"],
    "heart": ["corazón"], "alcohol": ["bebida alcohólica", "alcohol"],
}
# Coherencia semántica: la descripción del modelo (nombre + Cap3D) tiene que parecerse a la categoría. Las
# anotaciones de LVIS y los nombres de Sketchfab tienen ruido ("steering_wheel" con una válvula industrial).
COHERENCIA_MINIMA = 0.45
PESO_COHERENCIA = 8          # en el orden: que sea lo pedido pesa más que la popularidad
PREVIOS_POR_CATEGORIA = 150   # los mejores por calidad que se miden con embeddings (para no codificar 400 mil textos)
CATEGORIAS_SKETCHFAB_EXCLUIDAS = {"people", "characters-creatures"}   # escaneos de personas


def palabras_de(texto):
    return " " + re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip() + " "


def contiene(texto, frase):
    return f" {frase} " in texto


def calidad(a):
    detalle = min(max(math.log10(max(a["caras"], 1)), 3.5), 6) - 3.5
    return 2 * a["destacado"] + math.log1p(a["likes"]) + 0.3 * math.log1p(a["vistas"]) + 0.8 * detalle


def candidatos(rutas):
    """Recorre los 160 archivos de metadatos y se queda con lo que pasa los filtros (compacto, para no llenar la RAM)."""
    for archivo in sorted(glob.glob(str(METADATOS / "metadata" / "*.json.gz"))):
        for uid, a in json.load(gzip.open(archivo)).items():
            glb = (a.get("archives") or {}).get("glb") or {}
            caras = glb.get("faceCount") or a.get("faceCount") or 0
            if (a.get("license") not in LICENCIAS or a.get("isAgeRestricted") or uid not in rutas
                    or not glb.get("size") or glb["size"] > MAX_MB * 2**20 or not CARAS[0] <= caras <= CARAS[1]
                    or {c.get("slug", c.get("name")) for c in a.get("categories") or []} & CATEGORIAS_SKETCHFAB_EXCLUIDAS):
                continue
            usuario = a.get("user") or {}
            yield uid, {
                "nombre": (a.get("name") or "").strip(), "licencia": a["license"], "autor": usuario.get("displayName") or usuario.get("username") or "",
                "autor_url": usuario.get("profileUrl") or "", "url": a.get("viewerUrl") or "",
                "tamano": glb["size"], "caras": caras, "likes": a.get("likeCount") or 0,
                "vistas": a.get("viewCount") or 0, "destacado": bool(a.get("staffpickedAt")),
            }


def nombres_wordnet(categoria):
    from nltk.corpus import wordnet as wn
    if categoria in NOMBRES_LVIS:
        return NOMBRES_LVIS[categoria]
    base, _, aclaracion = categoria.partition("_(")
    aclaracion = aclaracion.rstrip(")").replace("_", " ")
    sentidos = wn.synsets(base, pos="n")
    if aclaracion:   # "bat_(animal)": el sentido cuya definición o hiperónimos mencionan la aclaración
        def menciona(s):
            textos = [s.definition()] + [l for h in s.closure(lambda x: x.hypernyms()) for l in h.lemma_names()]
            return any(p in " ".join(textos).lower() for p in aclaracion.lower().split())
        sentidos = [s for s in sentidos if menciona(s)] or sentidos
    for s in sentidos[:2]:
        nombres = [l.replace("_", " ") for l in s.lemma_names("spa") if l[:1].islower()]
        if nombres:
            return sorted(set(nombres), key=nombres.index)[:5]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=50, help="tope de la descarga (se puede subir y volver a correr)")
    args = ap.parse_args()
    r = random.Random(20261029)

    base = HF + "/allenai/objaverse/resolve/main/"
    for nombre in ["object-paths.json.gz", "lvis-annotations.json.gz"] + \
                  [f"metadata/{i // 1000:03d}-{i % 1000:03d}.json.gz" for i in range(160)]:
        descargar(base + nombre, METADATOS / nombre)
    rutas = json.load(gzip.open(METADATOS / "object-paths.json.gz"))
    lvis = json.load(gzip.open(METADATOS / "lvis-annotations.json.gz"))
    de_lvis = {uid: cat for cat, uids in lvis.items() for uid in uids}

    print("filtrando metadatos…", flush=True)
    modelos = dict(candidatos(rutas))
    print(f"  {len(modelos):,} modelos pasan licencia, tamaño y detalle", flush=True)

    print("descripciones de Cap3D…", flush=True)
    cap3d = {}
    ruta_cap3d = METADATOS / "Cap3D_automated_Objaverse_full.csv"
    descargar(HF + "/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_full.csv", ruta_cap3d)
    with open(ruta_cap3d, encoding="utf-8", newline="") as f:
        for fila in csv.reader(f):
            if len(fila) >= 2 and fila[0] in modelos:
                cap3d[fila[0]] = fila[1]

    # Categorías: dominio por palabras en el nombre y la descripción (los tags tienen demasiado ruido: "gear" es
    # también "equipo"); LVIS por la anotación
    categorias, descriptor = defaultdict(list), {}
    textos = {uid: palabras_de(m["nombre"] + " " + cap3d.get(uid, "")) for uid, m in modelos.items()}
    for nombres, frases, excluir in DOMINIO:
        clave = f"dominio:{nombres[0]}"
        descriptor[clave] = ", ".join(frases[:3])
        for uid, t in textos.items():
            if any(contiene(t, f) for f in frases) and not any(contiene(t, e) for e in excluir):
                categorias[clave].append(uid)
    for uid, cat in de_lvis.items():
        if uid in modelos:
            categorias[f"lvis:{cat}"].append(uid)
            descriptor[f"lvis:{cat}"] = cat.replace("_", " ").replace("(", "").replace(")", "")
    for uids in categorias.values():
        uids.sort(key=lambda u: -calidad(modelos[u]))
        del uids[PREVIOS_POR_CATEGORIA:]

    # Coherencia: similitud entre la categoría y lo que el modelo dice ser
    print("coherencia con embeddings…", flush=True)
    sys.path.insert(0, str(RAIZ / "backend"))
    import numpy as np
    from retrieval import MODELO_EMBEDDINGS, cargar_embeddings
    codificador = cargar_embeddings(MODELO_EMBEDDINGS)
    claves = sorted(categorias)
    v_cat = dict(zip(claves, codificador.encode([f"a 3D model of a {descriptor[c]}" for c in claves],
                                                normalize_embeddings=True, batch_size=256)))
    uids = sorted({u for us in categorias.values() for u in us})
    v_mod = dict(zip(uids, codificador.encode([(modelos[u]["nombre"] + ". " + cap3d.get(u, ""))[:300] for u in uids],
                                              normalize_embeddings=True, batch_size=256)))
    coherencia = {}
    for clave, us in categorias.items():
        for u in us:
            coherencia[(clave, u)] = float(np.dot(v_cat[clave], v_mod[u]))
        us[:] = sorted((u for u in us if coherencia[(clave, u)] >= COHERENCIA_MINIMA),
                       key=lambda u: -(calidad(modelos[u]) + PESO_COHERENCIA * coherencia[(clave, u)]))
    print(f"  {len(uids):,} modelos medidos · quedan {sum(map(len, categorias.values())):,} coherentes", flush=True)

    # Nombres en español y sinónimos reservados para la evaluación
    nombres_cat, reservado = {}, {}
    for nombres, _, _ in DOMINIO:
        nombres_cat[f"dominio:{nombres[0]}"] = nombres
    for clave in sorted(categorias):
        if clave.startswith("lvis:"):
            nombres = nombres_wordnet(clave[5:])
            # el sinónimo reservado tiene que ser una palabra de uso común (WordNet trae "ursidae", "tabular")
            comunes = [i for i, n in enumerate(nombres) if i > 0 and zipf_frequency(n, "es") >= FRECUENCIA_EVAL]
            if comunes and r.random() < RESERVA_EVAL:
                reservado[clave] = nombres.pop(r.choice(comunes))
            nombres_cat[clave] = nombres

    # Rondas: el siguiente mejor de cada categoría, hasta el tope
    tope, usado, elegidos = args.gb * 2**30, 0, {}
    limite = {c: POR_CATEGORIA[c.split(":")[0]] for c in categorias}
    for ronda in range(max(POR_CATEGORIA.values())):
        agregados = 0
        for clave in sorted(categorias, key=lambda c: (not c.startswith("dominio"), c)):
            if ronda >= limite[clave]:
                continue
            for uid in categorias[clave]:
                if uid not in elegidos:
                    break
            else:
                continue
            categorias[clave].remove(uid)
            if usado + modelos[uid]["tamano"] > tope:
                continue
            elegidos[uid] = (clave, ronda)
            usado += modelos[uid]["tamano"]
            agregados += 1
        print(f"  ronda {ronda + 1}: +{agregados} · {len(elegidos):,} modelos · {usado / 2**30:.1f} GB", flush=True)
        if not agregados:
            break

    campos = ["uid", "categoria", "nombres_es", "nombre_en", "descripcion_en", "consulta_eval", "ruta", "tamano",
              "caras", "licencia", "autor", "autor_url", "url", "calidad", "coherencia", "ronda"]
    with open(SELECCION, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for uid, (clave, ronda) in sorted(elegidos.items(), key=lambda x: (x[1][0], x[1][1])):
            m = modelos[uid]
            w.writerow({"uid": uid, "categoria": clave, "nombres_es": "|".join(nombres_cat.get(clave, [])),
                        "nombre_en": m["nombre"], "descripcion_en": cap3d.get(uid, ""),
                        "consulta_eval": reservado.get(clave, ""), "ruta": rutas[uid], "tamano": m["tamano"],
                        "caras": m["caras"], "licencia": LICENCIAS[m["licencia"]], "autor": m["autor"],
                        "autor_url": m["autor_url"], "url": m["url"], "calidad": round(calidad(m), 3),
                        "coherencia": round(coherencia[(clave, uid)], 3), "ronda": ronda})
    sin_nombre = sum(1 for c in {c for c, _ in elegidos.values()} if not nombres_cat.get(c))
    print(f"\n{len(elegidos):,} modelos · {usado / 2**30:.1f} GB · {len({c for c, _ in elegidos.values()})} categorías "
          f"({sin_nombre} sin nombre en español) · {len(cap3d):,} con descripción de Cap3D")
    print(f"-> {SELECCION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
