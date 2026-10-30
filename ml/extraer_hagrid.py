"""Extrae los 21 landmarks de las manos de HaGRID (fotos reales de gestos tomadas de internet) y los
guarda con el mismo formato que la herramienta de captura, para entrenar y evaluar el MLP.

Dataset: muestras de HaGRID a 384p publicadas en Hugging Face: 31.833 imágenes
(https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p) y 254.661
(https://huggingface.co/datasets/cj-mills/hagrid-sample-250k-384p), con la misma estructura. HaGRID: Kapitanov, Makhlyarchuk y
Kvanchiani (https://github.com/hukenovs/hagrid), licencia CC BY-SA 4.0 (variante de HaGRID).
Las imágenes NO se guardan en el repositorio: solo los landmarks, como en la captura propia.

Cada foto pasa por el mismo modelo que usa el visor (frontend/vendor/mediapipe/hand_landmarker.task).
Cada mano detectada se asigna a la caja anotada que la contiene, y de ahí sale su etiqueta.
La persona es el user_id de HaGRID: permite evaluar con personas que el modelo no vio.

Uso (con el entorno de Python 3.12 que tiene MediaPipe):
    ml/.venv-mediapipe/Scripts/python.exe ml/extraer_hagrid.py <carpeta de la muestra o su .zip> [--procesos 4] [--limite N]
        [--salida ml/data/externo/hagrid.csv]
"""
import argparse
import csv
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "frontend" / "vendor" / "mediapipe" / "hand_landmarker.task"
SALIDA = Path(__file__).parent / "data" / "externo" / "hagrid.csv"

# Etiqueta de HaGRID → clase del proyecto. Los gestos que el visor no usa (llamar, cuatro, tres,
# rock…) y las manos sin gesto se vuelven "ninguno": son los ejemplos que NO deben disparar acciones.
MAPEO = {
    "palm": "palma_abierta", "stop": "palma_abierta",
    "fist": "puño",
    "one": "señalar",
    "ok": "pellizco",          # pulgar e índice en contacto: el pellizco más parecido que tiene HaGRID
    "like": "pulgar_arriba",
    "dislike": "pulgar_abajo",
    "peace": "victoria",
}
OTRAS = ["call", "four", "mute", "peace_inverted", "rock", "stop_inverted", "three", "three2",
         "two_up", "two_up_inverted", "no_gesture"]

COLUMNAS = (["clase", "persona", "mano", "aspecto", "etiqueta_hagrid", "imagen"]
            + [f"{e}{i}" for i in range(21) for e in "xyz"])

_detector = None
_zip = None   # en cada proceso, el zip abierto (si la muestra se lee sin descomprimir)


def _iniciar():
    global _detector
    from mediapipe.tasks.python import BaseOptions, vision
    _detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))


def _caja_que_contiene(cx, cy, cajas):
    """Índice de la caja anotada [x, y, ancho, alto] (normalizada) más cercana que contiene el punto."""
    mejor, distancia = None, None
    for i, (x, y, w, h) in enumerate(cajas):
        margen_x, margen_y = w * 0.15, h * 0.15
        if x - margen_x <= cx <= x + w + margen_x and y - margen_y <= cy <= y + h + margen_y:
            d = (cx - (x + w / 2)) ** 2 + (cy - (y + h / 2)) ** 2
            if distancia is None or d < distancia:
                mejor, distancia = i, d
    return mejor


def procesar(tarea):
    """Una imagen → lista de filas (una por mano detectada y asignada a una caja anotada)."""
    import mediapipe as mp
    ruta, anotacion = tarea
    if "::" in ruta:   # "muestra.zip::carpeta/foto.jpg": se lee del zip y se decodifica en memoria
        global _zip
        import cv2
        import numpy as np
        archivo, ruta = ruta.split("::")
        if _zip is None:
            _zip = zipfile.ZipFile(archivo)
        bgr = cv2.imdecode(np.frombuffer(_zip.read(ruta), np.uint8), cv2.IMREAD_COLOR)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
    else:
        imagen = mp.Image.create_from_file(ruta)
    aspecto = imagen.width / imagen.height
    resultado = _detector.detect(imagen)
    filas = []
    for puntos, lados in zip(resultado.hand_landmarks, resultado.handedness):
        cx = sum(p.x for p in puntos) / 21
        cy = sum(p.y for p in puntos) / 21
        i = _caja_que_contiene(cx, cy, anotacion["bboxes"])
        if i is None:
            continue
        etiqueta = anotacion["labels"][i]
        clase = MAPEO.get(etiqueta, "ninguno")
        # Mismo criterio que frontend/js/hands.js (el visor recibe el video espejado)
        mano = "derecha" if lados[0].category_name == "Left" else "izquierda"
        filas.append([clase, anotacion["user_id"][:12], mano, f"{aspecto:.4f}", etiqueta, Path(ruta).stem]
                     + [f"{v:.5f}" for p in puntos for v in (p.x, p.y, p.z)])
    return filas


def tareas_zip(archivo: Path, limite):
    """Lo mismo que tareas(), leyendo el zip sin descomprimir: escribir medio millón de fotos al disco tarda horas."""
    z = zipfile.ZipFile(archivo)
    nombres = z.namelist()
    for etiqueta in list(MAPEO) + OTRAS:
        if etiqueta == "no_gesture":
            continue
        [json_anot] = [n for n in nombres if n.endswith(f"/ann_train_val/{etiqueta}.json")]
        anotaciones = json.loads(z.read(json_anot))
        imagenes = sorted(n for n in nombres if f"/train_val_{etiqueta}/" in n and n.endswith(".jpg"))[:limite]
        for n in imagenes:
            if Path(n).stem in anotaciones:
                yield f"{archivo}::{n}", anotaciones[Path(n).stem]


def tareas(base: Path, limite):
    if base.suffix == ".zip":
        yield from tareas_zip(base, limite)
        return
    # La carpeta de imágenes se llama según la muestra: hagrid_30k, hagrid_250k…
    [carpeta_fotos] = [c for c in base.glob("hagrid_*") if c.is_dir()]
    for etiqueta in list(MAPEO) + OTRAS:
        if etiqueta == "no_gesture":
            continue   # no tiene carpeta: aparece como la otra mano en las fotos de los demás gestos
        carpeta = carpeta_fotos / f"train_val_{etiqueta}"
        anotaciones = json.loads((base / "ann_train_val" / f"{etiqueta}.json").read_text(encoding="utf-8"))
        imagenes = sorted(carpeta.glob("*.jpg"))[:limite]
        for ruta in imagenes:
            if ruta.stem in anotaciones:
                yield str(ruta), anotaciones[ruta.stem]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, help="carpeta de la muestra (la que tiene hagrid_*/ y ann_train_val/) o su .zip")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--limite", type=int, default=None, help="imágenes por gesto (para probar)")
    args = ap.parse_args()

    lista = list(tareas(args.base, args.limite))
    print(f"{len(lista)} imágenes")
    salida = args.salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    total = sin_mano = 0
    with open(salida, "w", newline="", encoding="utf-8") as f, \
            ProcessPoolExecutor(args.procesos, initializer=_iniciar) as pool:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        for n, filas in enumerate(pool.map(procesar, lista, chunksize=32), 1):
            w.writerows(filas)
            total += len(filas)
            sin_mano += not filas
            if n % 10000 == 0:
                print(f"  {n}/{len(lista)} imágenes · {total} manos", flush=True)
    print(f"{total} manos de {len(lista)} imágenes ({sin_mano} sin mano asignada) -> {salida}")


if __name__ == "__main__":
    main()
