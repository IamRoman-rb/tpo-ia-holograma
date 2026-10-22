# Dataset de gestos y MLP

Protocolo completo en la sección 12 de [docs/propuesta.md](../docs/propuesta.md).

## 1. Grabar (cada integrante, con cámara)

1. Levantar el frontend (`cd frontend && python -m http.server 5500`) y abrir <http://localhost:5500/captura.html>.
2. Aceptar el consentimiento con un identificador anónimo (`p01`, `p02`, …).
3. Grabar los 5 gestos. La meta por defecto es 100 muestras por gesto, unos 7 segundos cada uno. Con 5 personas da 500 por clase, lo que pide la sección 12.2.
4. Variar la distancia, el ángulo, la mano y, si se puede, la luz.
5. Descargar el CSV y copiarlo a `ml/data/`.

| Clase | Gesto |
|---|---|
| `pellizco` | Pulgar e índice juntos |
| `palma_abierta` | Mano abierta frente a la cámara |
| `puño` | Mano cerrada |
| `señalar` | Índice extendido y el resto cerrado |
| `ninguno` | Mano relajada o movimientos al azar |

Formato de cada fila: `clase, persona, mano, aspecto, x0, y0, z0, …, x20, y20, z20`, con las coordenadas tal como las entrega MediaPipe. Solo se guardan coordenadas, nunca imágenes.

El escalado con dos manos no es una clase: el visor lo detecta cuando las dos manos están en pellizco.

## 2. Entrenar

```bash
python -m venv ml/.venv
ml/.venv/Scripts/activate            # Linux/Mac: source ml/.venv/bin/activate
pip install -r ml/requirements.txt
python -m pytest ml                  # la normalización de Python coincide con la del navegador
jupyter nbconvert --to notebook --execute --inplace ml/train_mlp.ipynb   # o abrirlo en Jupyter/VS Code
```

El notebook:
- evalúa **por persona**: cada persona se prueba con un modelo que no la vio;
- compara las reglas del visor con dos variantes de MLP;
- guarda gráficos y tablas en `ml/resultados/`;
- exporta `ml/gesture_model.onnx`.

Las filas grabadas en modo simulación (persona `simulacion`) se descartan automáticamente.
