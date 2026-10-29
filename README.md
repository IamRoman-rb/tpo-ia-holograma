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
| `scripts/` | Arranque con un solo comando, verificación pre-defensa y modo offline |

## Levantar el proyecto

Requisitos: Python 3.10+, Chrome o Edge actualizado, cámara web.

### Con un solo comando

Desde la raíz del repositorio:

```bash
python scripts/levantar.py
```

Levanta el backend (puerto 8000) y el frontend (puerto 5500) en la misma terminal y abre el visor en el navegador. La primera vez crea `backend/.venv` e instala las dependencias. Después solo las vuelve a instalar si cambia `backend/requirements.txt`. Ctrl+C apaga los dos.

Opciones:
- `--sin-voz`: no carga Whisper. Arranca más rápido y usa unos 3 GB menos de memoria.
- `--sin-navegador`: no abre el visor.
- `--recargar`: reinicia el backend al guardar cambios en el código.

Para levantarlos por separado, siguen los pasos de abajo.

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

### Holograma

Con el botón **Holograma** (o la tecla `H`) activado, que es lo que viene por defecto, el objeto se dibuja como un holograma sobre el fondo oscuro, con la cámara en su recuadro abajo a la derecha. Cada pieza tiene su color neón y se ve con sus aristas, una nube de puntos y un relleno tenue. A los costados aparecen etiquetas con el nombre y la función de cada pieza, con una línea guía hasta ella; al tocar una etiqueta se elige esa pieza. El catálogo pasa a ser una fila de pestañas abajo. **Paneles** muestra el panel de piezas y **Despiece** abre o cierra la vista explotada. Apagando **Holograma** se vuelve a la vista de estudio, con los materiales del modelo.

**Cámara de fondo** (o la tecla `C`, apagado por defecto) pone el video de la cámara a pantalla completa detrás del holograma: el esqueleto de la mano se dibuja sobre la mano real, el cursor de señalar queda en la punta del índice, al rotar con la palma aparece un aro y al escalar con las dos manos se ve el zoom.

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

### Vistas aparte: la red neuronal y el LLM

Dos páginas para ver cómo funcionan los modelos. Se abren con los botones **Red neuronal ↗** y **LLM ↗** de la barra, en otra pestaña o en otra pantalla. Reciben los datos del visor abierto en el mismo navegador.

- **[red.html](frontend/red.html)**: la mano que ve la cámara entra a la red y se ve cada capa:
  - las neuronas activas y las conexiones que más aportan;
  - la salida softmax contra el umbral, y las probabilidades en el tiempo;
  - la misma mano en la red, el árbol de decisión (con el camino de preguntas que siguió) y las reglas;
  - la arquitectura, los pasos explicados y la curva del algoritmo genético que la diseñó.

  Sin el visor abierto, recorre manos reales de HaGRID que la red no vio.
- **[llm.html](frontend/llm.html)**: el recorrido de cada pedido:
  - entrada, por texto o por voz con Whisper;
  - lo que respondió cada motor, con su distribución y su explicación (lo que escribió el LLM, las etiquetas BIO del transformer, las palabras que más pesaron en Naive Bayes);
  - el consenso con su fórmula y la decisión bayesiana de ejecutar o preguntar;
  - los filtros de seguridad y la ejecución.

  El **laboratorio** prueba una frase con el motor que se elija sin tocar el visor (también `llm.html?frase=…`).

Todas las técnicas, con sus resultados: [docs/tecnicas_ia.md](docs/tecnicas_ia.md).

### Pedidos sin LLM

Los pedidos escritos o por voz los interpreta el LLM (Claude). Sin Claude, los interpreta el **consenso de los motores propios** (`backend/consenso.py`): el transformer BETO ajustado, el LLM propio (Qwen ajustado), el intérprete con embeddings, Naive Bayes y las reglas votan, y la mejor respuesta sale de una combinación con pesos ajustados por máxima verosimilitud. El consenso acierta 305 de 309 frases en los siete sets de evaluación (97 % en la prueba, 98 % con objetos que nunca vio), pero tarda unos 5 s por pedido en el procesador. El motor solo más rápido y casi tan preciso es el transformer (300 de 309, 33 ms): se elige diciendo «usá el transformer». Uno de esos motores es un **modelo propio entrenado** que corre en el equipo (`backend/intent_modelo.py`): reconoce la acción con embeddings de la frase y un clasificador entrenado con 18.800 frases rioplatenses, y un etiquetador entrenado encuentra el objeto, de cualquier tipo, con lo que lo describe ("una silla de oficina con ruedas", "el cráneo de un tigre"). Aprendió con casi 10 mil nombres de objetos de WordNet en español. Con 45 pedidos de objetos que nunca vio acierta 91 %, contra 60 % de las reglas que usaba antes ([docs/metricas.md](docs/metricas.md#intérprete-de-pedidos-entrenado-respaldo-del-llm)). Si tampoco está ese modelo, quedan las reglas. El visor y el panel de actividad muestran qué motor respondió.

### Pedidos y comandos por voz

Con el backend levantado, la voz la reconoce **Whisper large-v3 en este equipo** (`backend/voz.py`): el audio no sale de la computadora.

- **Pedir por voz:** el botón del micrófono (o `V`) escucha un pedido y corta solo cuando dejás de hablar.
- **Manos libres:** el botón de los auriculares (o `M`) escucha sin parar y atiende lo que empieza con «holograma», por ejemplo «holograma, mostrame un engranaje». Con «holograma» solo, contesta «¿sí?» y espera el pedido.
- **Respuestas habladas:** el sistema contesta en voz alta qué hizo; `J` las apaga o las prende.

Por voz también se manejan los modelos:

- **Elegir motor:** «usá el árbol de decisión para los gestos», «pasá a la red neuronal», «usá Naive Bayes», «que responda el transformer», «volvé al modo automático».
- **Consultar:** «¿qué modelo respondió?», «¿qué tan seguro estás?», «¿por qué hiciste eso?».

Sin el backend, el micrófono usa el reconocimiento del navegador (Chrome y Edge, que mandan el audio a su servicio). Ver [docs/uso_responsable.md](docs/uso_responsable.md).

## Dataset de gestos y MLP

La red de gestos del visor (`frontend/ml/mlp_gestos.json`, versión 2) está entrenada con **HaGRID**, un dataset público de fotos de gestos: 276.237 manos de más de 23 mil personas, con aumento de datos. Con personas que no vio acierta 97,9 %, contra 62,4 % de las reglas, y confunde una mano sin gesto con un gesto solo el 1,3 % de las veces. Se probaron redes de hasta 724 mil parámetros, pero no mejoraron a la de 17 mil. Cómo se reproduce: [ml/README.md](ml/README.md#entrenar-con-hagrid-datos-de-internet). Resultados: [docs/metricas.md](docs/metricas.md).

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
