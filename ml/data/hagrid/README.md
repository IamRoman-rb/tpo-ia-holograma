# Landmarks de HaGRID

`hagrid.csv.xz` tiene los 21 puntos de la mano que MediaPipe extrajo (`ml/extraer_hagrid.py`) de la muestra de 254.661 fotos de HaGRID a 384p: 276.237 manos. Son solo coordenadas, no hay ninguna foto. Lo usan `ml/entrenar_red_gestos.py`, `ml/arboles/entrenar_arboles.py` y `ml/evolucion/neuroevolucion.py`, y pandas lo lee comprimido. Sin comprimir pesa 167 MB, más que el límite de GitHub; con xz, 44 MB.

Formato: `clase, persona, mano, aspecto, etiqueta_hagrid, imagen, x0, y0, z0, …, x20, y20, z20` (ver [ml/README.md](../../README.md)).

## Fuente y licencia

- **Dataset:** HaGRID, de Kapitanov, Makhlyarchuk y Kvanchiani: <https://github.com/hukenovs/hagrid>.
- **Muestra:** <https://huggingface.co/datasets/cj-mills/hagrid-sample-250k-384p>.
- **Licencia:** Creative Commons Attribution-ShareAlike 4.0 (variante de HaGRID). Estos landmarks son una obra derivada y se distribuyen con la misma licencia.

## Rehacerlo

```bash
ml/.venv-mediapipe/Scripts/python.exe -I ml/extraer_hagrid.py <carpeta>/hagrid-sample-250k-384p --procesos 11 --salida ml/data/hagrid/hagrid.csv
python -c "import lzma, shutil; shutil.copyfileobj(open('ml/data/hagrid/hagrid.csv','rb'), lzma.open('ml/data/hagrid/hagrid.csv.xz','wb', preset=9 | lzma.PRESET_EXTREME))"
```

Después hay que borrar `hagrid.csv`: si quedan los dos, se cargan las manos dos veces.
