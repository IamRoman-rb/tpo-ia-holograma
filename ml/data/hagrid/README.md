# Landmarks de HaGRID

`hagrid_*.csv.xz` tiene los 21 puntos de la mano que MediaPipe extrajo (`ml/extraer_hagrid.py`) de dos muestras de HaGRID a 384p: la de 254.661 fotos y la de 509.323. Las muestras se superponen: las fotos repetidas se sacan, así que cada mano aparece una vez. Son solo coordenadas, no hay ninguna foto.

Lo usan `ml/entrenar_red_gestos.py`, `ml/arboles/entrenar_arboles.py` y `ml/evolucion/neuroevolucion.py`: `preparar.cargar` une todos los archivos de la carpeta y pandas los lee comprimidos. Van partidos en archivos de menos de 95 MB porque GitHub no acepta archivos de más de 100 MB.

Formato: `clase, persona, mano, aspecto, etiqueta_hagrid, imagen, x0, y0, z0, …, x20, y20, z20` (ver [ml/README.md](../../README.md)).

## Fuente y licencia

- **Dataset:** HaGRID, de Kapitanov, Makhlyarchuk y Kvanchiani: <https://github.com/hukenovs/hagrid>.
- **Muestras:** <https://huggingface.co/datasets/cj-mills/hagrid-sample-250k-384p> y <https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p>.
- **Licencia:** Creative Commons Attribution-ShareAlike 4.0 (variante de HaGRID). Estos landmarks son una obra derivada y se distribuyen con la misma licencia.

## Rehacerlo

```bash
ml/.venv-mediapipe/Scripts/python.exe -I ml/extraer_hagrid.py <carpeta>/hagrid-sample-250k-384p --procesos 11 --salida ml/data/externo/hagrid.csv
ml/.venv-mediapipe/Scripts/python.exe -I ml/extraer_hagrid.py <carpeta>/hagrid-sample-500k-384p.zip --procesos 10 --salida ml/data/externo/hagrid500k.csv
backend/.venv/Scripts/python ml/empaquetar_hagrid.py ml/data/externo/hagrid.csv ml/data/externo/hagrid500k.csv
```

La segunda muestra se lee directo del zip, sin descomprimirla: escribir medio millón de fotos en el disco tarda horas. La extracción tarda ~1 h con 10 procesos.
