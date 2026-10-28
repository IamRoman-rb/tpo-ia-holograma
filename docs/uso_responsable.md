# Uso responsable

Responde a la expectativa superadora de **analizar limitaciones, sesgos y privacidad, y qué pasa cuando la IA se equivoca** (sección 16 de la [propuesta](propuesta.md)).

Para cada punto se indica qué está **implementado** (y dónde), y qué está **pendiente** de medir o hacer. Lo pendiente necesita personas o una cámara.

## Resumen

| Tema | Estado |
|---|---|
| Privacidad del video | ✅ Se procesa en el navegador y no se envía |
| Qué datos salen del equipo | ✅ Implementado y explicado en la ayuda del visor |
| Dataset de gestos | ✅ Consentimiento, identificador anónimo, solo coordenadas |
| Sesgos en la detección de manos | 🔶 Mitigaciones implementadas · medido con fotos de 1.134 personas (HaGRID): sin diferencia entre manos · ⏳ falta medirlo con la cámara |
| Pedidos por voz | ✅ Aviso de a dónde va el audio; solo se escucha con el micrófono encendido |
| Dataset de internet (HaGRID) | ✅ Licencia CC BY-SA 4.0, citado; solo se guardan coordenadas, nunca imágenes |
| Errores de la IA (sección 16.3) | ✅ Hay una protección implementada para cada caso de la tabla |
| Transparencia | ✅ Gesto, confianza, interpretación, motor y origen del modelo, todo en pantalla |
| Seguridad básica | ✅ Claves fuera del repositorio, límites de uso, validación de archivos |

---

## 1. Privacidad (sección 16.1)

### Qué se procesa y dónde

| Dato | Dónde se procesa | ¿Sale del equipo? | ¿Se guarda? |
|---|---|---|---|
| Video de la cámara | Navegador (MediaPipe, `frontend/js/hands.js`) | No | No |
| Puntos de la mano (21 por fotograma) | Navegador | No | Solo en la herramienta de captura, si la persona acepta |
| Frase del pedido | Backend local (`/intent`) | Sí, al LLM (Claude, de Anthropic), si hay clave configurada | No: el backend no registra las frases |
| Nombre del objeto a generar | Backend local (`/object`) | Sí, al servidor de generación en Colab y, si hay clave, al LLM para armar el prompt | Sí: queda en el catálogo como nombre del modelo generado |

- El video **nunca** se envía: en el código no hay ninguna llamada de red con imágenes. Se puede comprobar en las herramientas de desarrollador del navegador (pestaña Red): solo hay pedidos de texto a `localhost:8000`.
- La ayuda del visor (botón **?**) explica qué se procesa en el equipo y qué se envía a un servicio externo.
- Sin clave de API y con la generación desactivada, ningún dato sale del equipo: la intención la resuelve el intérprete entrenado (`backend/intent_modelo.py`), y si no está, el de reglas (`backend/intent.py`), los dos en el equipo y los objetos salen solo del catálogo.

### Pedidos por voz (`frontend/js/voz.js`)

- El micrófono se enciende solo al tocar el botón o apretar `V`, y se apaga al terminar el pedido.
- El reconocimiento lo hace el navegador. En Chrome y Edge el audio se envía a su servicio de voz (Google o Microsoft), salvo que el navegador ofrezca el reconocimiento en el equipo: en ese caso se usa, y el botón lo indica. Esto se explica en la ayuda del visor.
- El texto reconocido se trata igual que un pedido escrito: va al LLM (si está configurado) y pasa por las mismas validaciones.

### Dataset de internet (HaGRID)

- HaGRID fue publicado por sus autores (Kapitanov, Makhlyarchuk y Kvanchiani) con licencia CC BY-SA 4.0 (variante de HaGRID). Se usa la muestra pública de Hugging Face `cj-mills/hagrid-sample-30k-384p`.
- Las fotos quedan fuera del repositorio. Solo se guardan los 21 puntos de cada mano (`ml/data/externo/`, que tampoco se sube) y el modelo entrenado.
- El identificador de persona de HaGRID se usa solo para separar entrenamiento y prueba.

### Dataset de gestos (`frontend/captura.html`)

- **Consentimiento** explícito antes de empezar: explica qué se guarda y para qué.
- **Identificador anónimo**: el formulario no acepta espacios, para desalentar nombres reales.
- **Solo coordenadas**: 21 puntos por fotograma, nunca imágenes ni video.
- La persona descarga su propio archivo; no hay ningún envío automático.
- Las filas grabadas en **modo simulación** se marcan y el entrenamiento las descarta.

**Pendiente:** si alguien pide que se borren sus datos, eliminar su CSV de `ml/data/` y volver a entrenar.

---

## 2. Sesgos y equidad (sección 16.2)

### Detección de manos

MediaPipe puede rendir peor con ciertos tonos de piel, poca luz, manos con diferencias físicas (menos dedos, prótesis, artritis) o con guantes.

**Mitigaciones implementadas**

| Mitigación | Dónde |
|---|---|
| **Control alternativo**: botones y teclado para quien no pueda o no quiera usar gestos | `frontend/index.html`, `frontend/js/main.js` |
| Normalización por **tamaño de mano**: el mismo gesto da los mismos números en una mano chica o grande | `frontend/js/normalize.js` (probado con manos a distintas distancias) |
| Reglas basadas en **proporciones** entre puntos, no en distancias absolutas | `frontend/js/gestures.js` |
| Dataset con **varias personas**, ambas manos y variaciones de distancia, ángulo y luz | `frontend/captura.html` (sección 12.2) |
| Evaluación **por persona**: el resultado se informa para cada persona, así se ve si alguna queda peor | `ml/train_mlp.ipynb` |
| Se registra **qué mano** se usó en cada muestra | columna `mano` del dataset |

**Medido con fotos (HaGRID, 3.416 personas no vistas):** la red de gestos (versión 2) reconoce los gestos con la mano derecha (96,0 %) y con la izquierda (97,0 %) de forma parecida. De 790 personas con al menos 3 gestos, la mediana es 100 % y 27 (3,4 %) quedan por debajo del 80 %. La muestra de HaGRID no trae edad ni tono de piel, así que esos sesgos no se pudieron medir.

**Pendiente (fase 4): medir con personas reales.** Completar esta tabla con las pruebas de usuarios (sección 15.2), con 20 intentos por gesto en cada condición:

| Condición | Aciertos de gestos | Observaciones |
|---|---|---|
| Luz buena, mano derecha | | |
| Luz buena, mano izquierda | | |
| Luz baja | | |
| Fondo con mucho detalle | | |
| Persona que no participó del dataset | | |

### Interpretación del lenguaje (LLM)

- El prompt pide interpretar español rioplatense (`backend/prompts/intent_system.txt`).
- Las 30 frases de prueba (`eval/frases_prueba.csv`) incluyen voseo y formas coloquiales ("me mostrás…?", "girálo", "dale una vuelta").
- El intérprete por reglas falla justamente con formas naturales: 84 % en las frases de prueba y 68 % en 50 frases nuevas. **Pendiente:** medir el LLM con la misma batería cuando haya clave.
- El intérprete entrenado (`backend/intent_modelo.py`), que reemplaza a las reglas cuando no hay LLM, aprendió de frases generadas con plantillas escritas por una sola persona en español rioplatense, con los nombres de objetos de WordNet en español (que traen términos de España y de otros países: "ordenador", "coche"). Puede fallar con otras variantes del español o con formas que las plantillas no cubren. Con objetos de todo tipo que no vio acierta 91 % (objeto exacto); lo que más falla es confundir "mostrar" con "analizar una pieza" cuando el pedido usa "el/la". Su confianza es real (la probabilidad de la red): si es baja, el visor pide confirmar antes de ejecutar. Corre en el equipo, sin enviar la frase a ningún servicio.

### Generador 3D

Hereda los sesgos de sus datos de entrenamiento: tiende a generar la versión "típica" de un objeto. **Pendiente:** registrarlo al evaluar los generadores (`eval/generacion3d/`), con piezas industriales que no sean las más comunes.

---

## 3. Qué pasa cuando la IA se equivoca (sección 16.3)

| Error posible | Efecto | Protección implementada | Dónde |
|---|---|---|---|
| Gesto mal clasificado | Acción no deseada en el visor | Umbral de confianza de 0.8 con votación de los últimos 6 fotogramas; colores en el recuadro de cámara (ámbar = detectado pero no aplicado); **botón Deshacer** (tecla Z), que vuelve a como estaba antes del gesto | `gestures.js`, `overlay.js`, `historial.js`, `controles.js` |
| El LLM interpreta mal el pedido | Se muestra otro objeto | Se muestra lo entendido ("Entendí: mostrar «motor» · vista trasera · 93 %"); si la confianza es menor a 0.5, se pide confirmación antes de actuar | `ui.js`, `main.js`, `backend/safety.py` |
| La búsqueda devuelve una pieza equivocada | Se muestra un objeto parecido pero incorrecto | Umbral calibrado para preferir "no la tengo" antes que mostrar una pieza equivocada (0.62, `docs/decisiones.md`); se muestra el nombre de la pieza encontrada | `backend/retrieval.py` |
| Modelo 3D con geometría errónea | Información engañosa sobre el producto | Etiqueta visible **"Generado por IA · no apto para fabricación sin revisión"**, que se mantiene cuando el modelo se reutiliza; marca **IA** en el catálogo | `ui.js`, `backend/generate3d.py` |
| Pedido inapropiado o malicioso | Contenido no deseado | Filtro de objetos fuera del dominio, tanto en la intención como antes de generar; el prompt del LLM indica responder "desconocido" | `backend/safety.py`, `backend/main.py` |
| Inyección de instrucciones en la frase | El LLM obedece al usuario en lugar de las reglas | La salida está restringida a un JSON Schema con 6 acciones posibles y se valida con Pydantic: lo peor que puede pasar es que se muestre otro objeto | `backend/schemas.py`, `backend/safety.py` |

**Limitaciones conocidas**

- El filtro de dominio es una **lista de palabras**: se puede esquivar con sinónimos que no estén en la lista. Lo complementa el LLM, pero sin clave solo queda la lista.
- "Deshacer" cubre el estado del objeto (posición, rotación y escala), no el cambio de modelo: al cargar otro objeto el historial se vacía.
- El umbral de búsqueda se calibró con 20 consultas; hay que confirmarlo con un set nuevo (`docs/metricas.md`).

---

## 4. Transparencia (sección 16.4)

En pantalla, en todo momento:

- **Gesto detectado y confianza** (panel superior izquierdo), con el esqueleto de cada mano y su etiqueta en el recuadro de cámara.
- **Qué se entendió del pedido** y si lo interpretó el LLM o las reglas.
- **Origen del modelo**: "Del catálogo" o "Generado por IA · no apto para fabricación sin revisión".
- **"Simulación"**, cuando las manos no vienen de una cámara real.
- **Actividad de la IA** (botón o tecla `L`): qué ve la red neuronal en cada cuadro (activaciones, probabilidades contra el umbral) y qué pasó con cada pedido (si lo interpretó el LLM o las reglas y por qué, qué respondió, qué cambiaron los filtros de seguridad, cuánto tardó y si el objeto salió del catálogo o se generó). Se puede abrir en otra ventana para mostrarlo a la audiencia. Los datos pasan entre pestañas del mismo navegador (BroadcastChannel) y no salen del equipo.

### Qué hace y qué no hace el sistema

| Hace | No hace |
|---|---|
| Mostrar un objeto aproximado para conversar con un cliente | Garantizar medidas ni geometría exactas |
| Generar un modelo cuando no existe en el catálogo | Reemplazar el modelado de un diseñador para fabricar |
| Manipular el objeto con 4 gestos (pellizco, palma, puño, dos manos) | Reconocer cualquier movimiento de la mano |
| Interpretar pedidos en español | Responder preguntas ajenas a mostrar objetos |

---

## 5. Seguridad básica (sección 16.5)

| Medida | Dónde |
|---|---|
| Claves de API en `.env`, que está en `.gitignore`; nunca en el repositorio | `.env.example`, `.gitignore` |
| Límite de 30 pedidos por minuto por cliente | `backend/main.py` |
| Límite de 5 generaciones cada 10 minutos (usa GPU y puede tener costo) | `backend/main.py` |
| Frases de hasta 300 caracteres | `backend/schemas.py` |
| Validación de cada `.glb` generado: cabecera glTF 2.0, largo declarado y tope de 20 MB | `backend/generate3d.py` |
| El servidor de generación exige una contraseña, porque su URL es pública | `backend/generador_colab/servidor_generacion.ipynb` |
| CORS limitado al frontend local | `backend/main.py` |
| Librerías de terceros con hash verificado | `frontend/vendor/VERSIONES.md`, `scripts/check_all.sh` |

---

## Pendientes

1. **Medir sesgos de detección** con personas diversas y distintas condiciones de luz: completar la tabla de la sección 2.
2. **Medir el LLM** con las 30 frases, incluida la de inyección de instrucciones.
3. **Registrar los sesgos del generador 3D** durante su evaluación.
4. **Confirmar el umbral de búsqueda** con un segundo set de consultas.
