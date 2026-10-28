# TPO IA — Visor 3D controlado con gestos y generado por IA

> **Un visor 3D que se controla con las manos y se crea con una frase, para que los equipos comerciales y técnicos muestren productos o piezas a un cliente sin mouse ni modelado 3D.**

Inteligencia Artificial · UADE · 2do cuatrimestre 2026. La propuesta completa está en [docs/propuesta.md](docs/propuesta.md).

Arquitectura (entregable 4): [docs/arquitectura.md](docs/arquitectura.md), con el diagrama editable en [docs/arquitectura.drawio](docs/arquitectura.drawio).

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/` | Propuesta, arquitectura, métricas, uso responsable, decisiones y presentación |
| `frontend/` | Visor Three.js + MediaPipe en el navegador |
| `backend/` | API FastAPI: intención (LLM), búsqueda por similitud, generación 3D, seguridad |
| `ml/` | Captura del dataset de gestos y entrenamiento del MLP |
| `eval/` | Frases de prueba del LLM y resultados de pruebas con usuarios |
| `scripts/` | Verificación pre-defensa y modo offline |

## Levantar el proyecto

Requisitos: Python 3.10+, Chrome o Edge actualizado, cámara web.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env       # opcional: ANTHROPIC_API_KEY para usar el LLM
uvicorn main:app --reload --port 8000
```

Verificar en <http://localhost:8000/health>. La primera vez descarga el modelo de embeddings (unos 500 MB, con PyTorch incluido en la instalación). Sin `ANTHROPIC_API_KEY`, los pedidos los interpreta un intérprete por reglas de respaldo.

### Frontend

```bash
cd frontend
python -m http.server 5500
```

Abrir <http://localhost:5500> y permitir el acceso a la cámara. El frontend funciona aunque el backend no esté levantado: los pedidos se buscan en el catálogo local.

## Antes de la defensa

```bash
bash scripts/check_all.sh --tests
```

El script revisa los servicios, el plan B sin internet, el catálogo, la configuración y las pruebas. Las librerías del frontend están en `frontend/vendor/`, así que el visor funciona sin internet. Ver [scripts/demo_offline.md](scripts/demo_offline.md).

El guion de los 15 minutos, la demo paso a paso y las respuestas a las preguntas probables están en [docs/presentacion/guion.md](docs/presentacion/guion.md).

## Uso

| Gesto | Acción |
|---|---|
| Pellizco (pulgar + índice) | Tomar y mover el objeto |
| Palma abierta | Rotar siguiendo la mano |
| Pellizco con las dos manos | Separar para agrandar, juntar para achicar |
| Señalar (solo el índice) | Mover el cursor; apuntar 0,8 s a una pieza la elige |
| Pellizco sobre la pieza elegida | Sacar solo esa pieza y moverla |
| Victoria (índice y medio en V) | Analizar la pieza elegida; sin pieza elegida, abrir o cerrar el despiece |
| Puño | Salir del análisis; si no se está analizando, reiniciar la vista |
| Pulgar arriba | Confirmar un pedido |
| Pulgar abajo | Deshacer el último cambio |

Los gestos los reconoce una red neuronal (MLP) entrenada con fotos reales de miles de personas, o las reglas geométricas: se elige en el panel de la izquierda (**Clasificador**).

### Holograma sobre la cámara

Con el botón **Holograma** (o la tecla `H`) activado, que es lo que viene por defecto, el video de la cámara ocupa toda la pantalla y el objeto se dibuja encima como un holograma. Cada pieza tiene su color neón y se ve con sus aristas, una nube de puntos y un relleno tenue. A los costados aparecen etiquetas con el nombre y la función de cada pieza, con una línea guía hasta ella; al tocar una etiqueta se elige esa pieza. El esqueleto de la mano se dibuja sobre la mano real, el cursor de señalar queda en la punta del índice, al rotar con la palma aparece un aro y al escalar con las dos manos se ve el zoom. El catálogo pasa a ser una fila de pestañas abajo. **Paneles** muestra el panel de piezas y **Despiece** abre o cierra la vista explotada. Sin cámara, el holograma se ve sobre el fondo oscuro y el recuadro de cámara sigue abajo a la derecha. Apagando **Holograma** se vuelve a la vista de estudio, con los materiales del modelo.

### Actividad de la IA en vivo

**Actividad IA** (o la tecla `L`) abre un panel que muestra qué hace cada modelo mientras se usa el visor:

- **Red neuronal de gestos**: las activaciones de sus cuatro capas, de la entrada (los 21 puntos de la mano) a la salida (las 8 clases), con líneas entre las neuronas más activas. Debajo están la probabilidad de cada clase contra el umbral de 90 % y una línea de tiempo con el gesto aplicado en los últimos segundos. Con las reglas (y en la simulación) se ven las métricas de la mano contra sus umbrales.
- **LLM**: cada pedido desde que sale hasta que se ejecuta. Se ve qué motor lo interpretó (el modelo de Claude o, si no está disponible, las reglas y por qué), los tokens, el tiempo del LLM y el total, el JSON que devolvió y si los filtros lo cambiaron, si pidió confirmación, y si el objeto salió del catálogo (con su similitud) o se generó.

**↗ Ventana** abre lo mismo en otra ventana ([actividad.html](frontend/actividad.html)), para mostrarlo en otra pantalla durante la demo. Los datos pasan entre las pestañas del mismo navegador y no salen del equipo.

### Desarmar el objeto

Para **analizar una pieza**: señalala un momento (o tocala en la lista o en el modelo) y hacé la victoria, apretá `I` o el botón **Analizar**. Queda sola, centrada y grande, con un panel que dice qué es, de qué material y cuánto mide; se gira con la palma. El puño, `Esc` o "volvé al conjunto" la devuelven a su lugar. También se puede pedir: "mostrame solo el volante".

Los modelos están hechos por piezas. El panel **Piezas** las lista en el orden en que se desmontan: **Desmontar** saca la próxima, **Montar** vuelve a poner la última, **Armar todo** las junta y la barra **Vista explotada** las separa de a poco. Al tocar una pieza en la lista o en el modelo, se resalta y se muestra para qué sirve. También funciona con el teclado (`X` desmonta, `A` arma) y con pedidos escritos: "mostrame la válvula desarmada", "sacale una pieza", "armalo de nuevo". Deshacer también vuelve atrás el desarmado.

El recuadro de cámara (abajo a la derecha) muestra el esqueleto de cada mano y una etiqueta con el gesto que se detecta. **Cian** significa que el gesto se está aplicando al objeto y **ámbar** que se reconoce, pero todavía no alcanza la confianza mínima. El botón ⤢ agranda el recuadro. Si no hay cámara, **Simular manos** reproduce un recorrido con manos sintéticas (palma, pellizco, dos manos y puño) que pasa por el mismo clasificador. Sirve para probar y mostrar el sistema, pero no reemplaza las pruebas con manos reales.

Sin cámara se puede usar el teclado (flechas para rotar, `+` y `−` para escalar, `R` para reiniciar) o los botones de la pantalla. **Deshacer** (botón o tecla `Z`) vuelve a como estaba el objeto antes del último cambio, por ejemplo si un gesto se detectó mal.

Qué datos se procesan en el equipo y cuáles salen de él, sesgos y protecciones: [docs/uso_responsable.md](docs/uso_responsable.md). Con la tecla `D` se abre el panel de depuración, que muestra las métricas de la mano en vivo y sirve para calibrar los umbrales de `frontend/js/gestures.js`.

### Pedidos por voz

El botón del micrófono (o la tecla `V`) escucha un pedido en español y lo trata igual que uno escrito. Si el pedido vino por voz, la respuesta también se escucha. Funciona en Chrome y Edge; en Firefox queda solo el texto. En esos navegadores el audio lo procesa su servicio de voz, salvo que puedan reconocerlo en el equipo (ver [docs/uso_responsable.md](docs/uso_responsable.md)).

## Dataset de gestos y MLP

El MLP del visor (`frontend/ml/mlp_gestos.json`) está entrenado con **HaGRID**, un dataset público de fotos de gestos de miles de personas: 97,4 % de exactitud con personas que no vio, contra 63,2 % de las reglas. Cómo se reproduce: [ml/README.md](ml/README.md#entrenar-con-hagrid-datos-de-internet). Resultados: [docs/metricas.md](docs/metricas.md).

La página <http://localhost:5500/captura.html> sirve para grabar un dataset propio con la cámara. Los pasos para grabar y entrenar están en [ml/README.md](ml/README.md).

## Generación 3D

Un pedido que no está en el catálogo se puede generar de dos formas (variable `GEN3D_MODO` en `.env`):

- **Por piezas con el LLM** (`ensamble`, la opción por defecto si hay `ANTHROPIC_API_KEY`). Claude diseña el objeto como un ensamble de formas simples (cilindros, conos, tubos, engranajes, cajas…) con medidas, materiales, orden de desmontaje y dirección de salida de cada pieza. El backend valida ese diseño y lo convierte en un `.glb` con una pieza por nodo, así que lo generado se puede desarmar igual que el catálogo. No necesita GPU. El prompt está en `backend/prompts/ensamble_system.txt`.
- **TripoSR o Shap-E en Colab** (`remoto`). Dan formas más orgánicas, pero en una sola malla: no se pueden desarmar. La evaluación se corre en Google Colab, porque necesita GPU. Los pasos y la rúbrica de calidad están en [eval/generacion3d/README.md](eval/generacion3d/README.md).

En los dos casos, lo generado se guarda en `frontend/models/generados/` y queda en el catálogo para la próxima vez.

Para usar Colab, se corre `backend/generador_colab/servidor_generacion.ipynb` en Colab y se copian a `.env` las tres líneas que imprime al final (`GEN3D_MODO`, `GEN3D_URL` y `GEN3D_TOKEN`). Sin clave de API ni servidor, los pedidos se resuelven solo con el catálogo.

## Catálogo, pruebas y evaluación

```bash
python scripts/generar_modelos_demo.py   # regenera frontend/models/*.glb y catalogo.json
node frontend/tests/gestos.test.mjs      # pruebas de normalización, gestos y búsqueda local
node frontend/tests/piezas.test.mjs      # orden y recorrido del desarmado, análisis de una pieza
node frontend/tests/mlp.test.mjs         # el MLP en JavaScript da lo mismo que en Python
node frontend/tests/voz.test.mjs         # pedidos por voz (con un reconocimiento falso)
node frontend/tests/holograma.test.mjs   # nube de puntos, etiquetas a los costados y dedo en la pantalla
python -m pytest ml                      # la normalización de Python coincide con la del navegador (entorno de ml/)
cd backend && python -m pytest tests     # pruebas del backend (con el entorno virtual activo)
python eval/eval_intent.py               # aciertos de intención: reglas vs. LLM (desde la raíz)
python eval/eval_busqueda.py             # búsqueda en el catálogo y calibración del umbral
python eval/eval_ensamble.py             # generación por piezas con el LLM (tiene costo; --prueba para probar sin API)
```

Los resultados están en [docs/metricas.md](docs/metricas.md) y las decisiones técnicas en [docs/decisiones.md](docs/decisiones.md).

## Ramas

| Rama | Uso |
|---|---|
| `main` | Versión estable y presentable. Es la que se usa en la defensa. |
| `test` | Integración y pruebas antes de pasar a `main`. |
| `desarrollo` | Rama de trabajo diario donde se unen las funciones terminadas. |

Flujo: `feature/*` → `desarrollo` → `test` → `main`.

## Convenciones de trabajo

- Rama `main` protegida; una rama por función creada desde `desarrollo` (`feature/gestos-mlp`, `feature/llm-intent`).
- *Pull request* con al menos una revisión de otro integrante antes de unir.
- Nunca subir claves de API: van en `.env`, que está en `.gitignore`.
- Las decisiones técnicas se registran en [docs/decisiones.md](docs/decisiones.md).
- Los prompts del producto viven en `backend/prompts/`, no dentro del código.
