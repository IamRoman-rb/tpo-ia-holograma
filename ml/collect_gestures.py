"""Captura del dataset de gestos (protocolo de la sección 12).

Uso:  python collect_gestures.py --clase pellizco --persona p01
Guarda una fila por fotograma en data/<clase>.csv:
    clase, persona, x0, y0, z0, ..., x20, y20, z20
Solo se guardan landmarks (coordenadas), nunca imágenes.
Teclas: espacio = grabar/pausar, q = salir.
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

CLASES = ["pellizco", "palma_abierta", "puño", "señalar", "escala_dos_manos", "ninguno"]
DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clase", required=True, choices=CLASES)
    ap.add_argument("--persona", required=True, help="id anónimo, ej. p01")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    ruta = DATA / f"{args.clase}.csv"
    nuevo = not ruta.exists()

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    grabando, n = False, 0

    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["clase", "persona"] + [f"{e}{i}" for i in range(21) for e in "xyz"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if grabando and res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                w.writerow([args.clase, args.persona] + [v for p in lm for v in (p.x, p.y, p.z)])
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
