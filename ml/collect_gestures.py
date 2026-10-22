"""Captura del dataset de gestos desde Python (alternativa a frontend/captura.html).

La forma recomendada es la página frontend/captura.html: no requiere instalar nada y
usa el mismo detector que el visor. Este script queda para quien prefiera OpenCV.

Uso:  python ml/collect_gestures.py --clase pellizco --persona p01
Agrega una fila por fotograma a ml/data/<persona>_python.csv con el mismo formato:
    clase, persona, mano, aspecto, x0, y0, z0, ..., x20, y20, z20
Solo se guardan landmarks (coordenadas), nunca imágenes.
Teclas: espacio = grabar/pausar, q = salir.
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

CLASES = ["pellizco", "palma_abierta", "puño", "señalar", "ninguno"]
DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clase", required=True, choices=CLASES)
    ap.add_argument("--persona", required=True, help="id anónimo, ej. p01")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    ruta = DATA / f"{args.persona}_python.csv"
    nuevo = not ruta.exists()

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    grabando, n = False, 0

    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["clase", "persona", "mano", "aspecto"] + [f"{e}{i}" for i in range(21) for e in "xyz"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            alto, ancho = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if grabando and res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                # MediaPipe supone imagen espejada: "Left" es la mano derecha del usuario
                etiqueta = res.multi_handedness[0].classification[0].label
                mano = "derecha" if etiqueta == "Left" else "izquierda"
                w.writerow([args.clase, args.persona, mano, round(ancho / alto, 4)]
                           + [round(v, 5) for p in lm for v in (p.x, p.y, p.z)])
                n += 1
            estado = f"{'GRABANDO' if grabando else 'pausa'} | {args.clase} | {n} muestras"
            cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("captura", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(" "):
                grabando = not grabando
            elif tecla == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{n} muestras agregadas a {ruta}")


if __name__ == "__main__":
    main()
