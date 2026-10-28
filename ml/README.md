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

## Entrenar con HaGRID (datos de internet)

La red de gestos del visor se entrenó con **HaGRID** (Kapitanov, Makhlyarchuk y Kvanchiani, <https://github.com/hukenovs/hagrid>), licencia CC BY-SA 4.0 (variante de HaGRID).

**Versión 2 (la actual):** muestra de 254.661 fotos a 384p, <https://huggingface.co/datasets/cj-mills/hagrid-sample-250k-384p> (6,8 GB). Se entrena con PyTorch en el mismo entorno (`ml/.venv-mediapipe/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`):

```bash
ml/.venv-mediapipe/Scripts/python.exe -I ml/extraer_hagrid.py <carpeta>/hagrid-sample-250k-384p --procesos 11   # ~35 min
ml/.venv-mediapipe/Scripts/python.exe ml/entrenar_red_gestos.py      # ~10 min en CPU; --rapido para probar el script
```

`entrenar_red_gestos.py` compara cinco tamaños de red con aumento de datos, elige con validación, evalúa las reglas (con node) y la red anterior sobre la misma prueba, y escribe `frontend/ml/mlp_gestos.json`, `frontend/tests/datos/mlp_casos.json` y `ml/resultados/hagrid_red_v2_reporte.json`.

**Versión 1 (para reproducirla):** muestra de 31.833 fotos, <https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p> (1,0 GB), con scikit-learn. Sus landmarks van en `ml/data/externo/v1_30k/`:

1. Descargar y descomprimir el zip **fuera del repositorio** (las fotos no se suben).
2. Crear el entorno con Python 3.12, porque MediaPipe no tiene paquetes para 3.13:
   ```bash
   py -3.12 -m venv ml/.venv-mediapipe
   ml/.venv-mediapipe/Scripts/python.exe -m pip install mediapipe pandas scikit-learn
   ```
3. Extraer los landmarks (unos 15 minutos con 6 procesos). Usa el mismo modelo de MediaPipe que el visor y guarda solo coordenadas en `ml/data/externo/v1_30k/hagrid_30k.csv`:
   ```bash
   ml/.venv-mediapipe/Scripts/python.exe ml/extraer_hagrid.py <carpeta>/hagrid-sample-30k-384p --procesos 6 --salida ml/data/externo/v1_30k/hagrid_30k.csv
   ```
4. Separar por persona y evaluar las reglas sobre la prueba (la partición sale de `preparar.particion`):
   ```bash
   ml/.venv-mediapipe/Scripts/python.exe -I -c "import sys; sys.path.insert(0,'ml'); import preparar as p; d=p.cargar('ml/data/externo/v1_30k'); d['part']=d.persona.map(p.particion); c=[x for x in d.columns if x not in ('archivo','fila','part')]; import os; os.makedirs('ml/data/externo/prueba',exist_ok=True); d[d.part=='prueba'][c].to_csv('ml/data/externo/prueba/hagrid_prueba.csv',index=False)"
   node ml/evaluar_reglas.mjs ml/data/externo/prueba ml/resultados/hagrid_reglas_prueba.csv
   ```
5. Entrenar, comparar y exportar el modelo para el navegador:
   ```bash
   ml/.venv-mediapipe/Scripts/python.exe ml/entrenar_mlp.py
   ```
   Elige la arquitectura y el umbral con validación, informa la prueba (reglas vs. MLP) y escribe `frontend/ml/mlp_gestos.json`, `ml/resultados/hagrid_mlp_reporte.json` y los casos de `frontend/tests/datos/mlp_casos.json`.

Para calibrar umbrales de las reglas con datos (solo con la partición de entrenamiento): `node ml/explorar_metricas.mjs <csv>`.

Correspondencia de clases: palm/stop → palma abierta · fist → puño · one → señalar · ok → pellizco (el más parecido que tiene HaGRID) · like → pulgar arriba · dislike → pulgar abajo · peace → victoria · el resto → ninguno.

## Intérprete de pedidos entrenado (lenguaje)

El respaldo del LLM cuando no hay clave o no hay internet (`backend/intent_modelo.py`). Se entrena con el entorno del backend, que ya tiene sentence-transformers y scikit-learn:

```bash
backend/.venv/Scripts/python ml/intencion/generar_frases.py      # 4.400 frases de plantilla -> ml/intencion/frases_entrenamiento.csv
backend/.venv/Scripts/python ml/intencion/entrenar_intencion.py  # elige embeddings y clasificador, exporta backend/modelos/intencion.json
backend/.venv/Scripts/python eval/eval_intent.py --frases nuevas # de punta a punta, contra las reglas (y el LLM si hay clave)
```

Las plantillas no usan los objetos de las frases de prueba y se saltean las frases de prueba y de validación (y sus variantes con muletillas). Resultados y por qué cada set mide algo distinto: [docs/metricas.md](../docs/metricas.md#intérprete-de-pedidos-entrenado-respaldo-del-llm).
