# Prompt para Claude: informe técnico + guion de diapositivas

> Copiá todo lo que está debajo de la línea y pegalo en Claude (claude.ai, con el modelo más grande disponible).
> Los datos están verificados contra el repositorio al 11/11/2026. Si cambia algo antes de la defensa, corregí la ficha.

---

## Tu tarea

Sos un ingeniero de IA que tiene que explicarle un proyecto universitario a su propio equipo, la noche antes de la defensa. Con la **ficha técnica** de abajo (es la única fuente de verdad), generá tres cosas:

1. **Un informe técnico completo** para que lo lean mis compañeros. Tiene que ser bien técnico (nombres de modelos, arquitecturas, parámetros, métricas, tamaños, tiempos), pero que se entienda: cada técnica se explica primero en una o dos frases simples ("qué problema resuelve y cómo, en criollo") y después con el detalle. Usá analogías cortas cuando ayuden. Incluí tablas donde haya números.
2. **Un guion de diapositivas para PowerPoint**, para una exposición de **15 minutos** (objetivo: 13 a 14). Entre 14 y 18 diapositivas. Para cada una: título, 3 a 5 viñetas cortas (no párrafos), qué visual poner (diagrama, tabla, captura, gráfico, con qué datos), notas del orador (lo que se dice, 60–120 palabras) y el tiempo estimado. Incluí una diapositiva para la **demo en vivo** con el paso a paso y qué hacer si algo falla.
3. **Preguntas probables del jurado** (al menos 12) con una respuesta corta y honesta para cada una, y un **glosario** de los términos técnicos (MLP, LoRA, BIO, Viterbi, ECE, McNemar, etc.).

Reglas:
- **No inventes números ni funcionalidades.** Si algo no está en la ficha, no lo pongas. Si una métrica está marcada como pendiente o no medida, decilo así.
- Respetá las aclaraciones de la sección "Estado actual y advertencias": hay documentación vieja y un modelo que todavía se está entrenando.
- Escribí en español rioplatense, claro y directo. Nada de relleno ni frases de marketing.
- Las cifras van con coma decimal (98,1 %) y punto de miles (109.524).

---

## Ficha técnica del proyecto

### 1. Qué es

**Visor 3D que se controla con las manos y se crea con una frase.** Es un TPO de Inteligencia Artificial (UADE, 2.º cuatrimestre 2026). Está pensado para que equipos comerciales y técnicos muestren un producto o una pieza a un cliente sin mouse y sin saber modelar en 3D.

- El usuario pide un objeto, escribiendo o por voz ("mostrame un motor V8 desarmado").
- El sistema entiende el pedido y busca el modelo 3D en un catálogo de miles de modelos. Si no lo encuentra, lo genera con IA.
- El objeto se muestra como un **holograma** (fondo oscuro, piezas en neón, aristas, nube de puntos y etiquetas con el nombre y la función de cada pieza).
- Se manipula con **gestos de la mano frente a la cámara web**: mover, rotar, escalar, elegir una pieza, sacarla, desarmar todo el objeto, analizar una pieza, deshacer.
- Todo lo que hace la IA se ve en vivo: páginas que muestran la red neuronal capa por capa y el recorrido de cada pedido por los modelos de lenguaje.

### 2. Arquitectura

**Tres bloques:**

| Bloque | Tecnología | Qué hace |
|---|---|---|
| Frontend (navegador) | HTML + JavaScript, **Three.js** (3D), **MediaPipe** (manos) | Cámara, gestos, visor, holograma, desarme, paneles. El video **nunca sale de la computadora**. Funciona sin internet: las librerías están dentro del proyecto (`frontend/vendor/`, 27 MB) |
| Backend local | Python, **FastAPI** (puerto 8000) | Interpretar pedidos, buscar en el catálogo, generar 3D, reconocer voz, filtros de seguridad |
| Externos (opcionales) | Claude (API de Anthropic), Google Colab con GPU | Interpretar pedidos y diseñar objetos por piezas; generar mallas con TripoSR o Shap-E |

Se levanta todo con un solo comando: `python scripts/levantar.py` (backend en el 8000, frontend en el 5500, y abre el navegador).

**API del backend:**

| Ruta | Qué hace |
|---|---|
| `GET /health` | Estado, tipo de búsqueda, cantidad de modelos, generador activo |
| `POST /intent` | Frase → JSON `{accion, objeto, parametros, confianza, requiere_confirmacion, motor}` |
| `POST /object` | Objeto → modelo 3D (`origen`, `modelo_url`, `similitud`, `metodo`), o lo genera |
| `GET /catalog` | Lista del catálogo |
| `POST /voz` | Audio → texto con Whisper local |

**Flujo 1, gestos (todo en el navegador):** cámara → MediaPipe da 21 puntos 3D por mano → filtro de Kalman saca el temblor → normalización (origen en la muñeca, escala según el tamaño de la mano) → clasificador del gesto (red neuronal, árbol de decisión o reglas, a elección) → si la confianza llega al umbral (80 % por defecto, ajustable de 50 a 95 %), se aplica al objeto. Antes de cada gesto se guarda el estado para poder deshacer.

**Flujo 2, pedidos:** frase escrita o por voz → `POST /intent` → la cadena de intérpretes la convierte en un JSON con esquema fijo → filtros de seguridad → si la confianza es baja, pide confirmación → `POST /object` → búsqueda híbrida en el catálogo → si está, lo devuelve; si no, lo genera, lo guarda en el catálogo y lo devuelve.

**Si algo falla, nunca se cae:** sin backend, el visor busca en el catálogo local; sin Claude, interpreta el consenso de modelos propios; sin esos, las reglas; sin cámara, quedan los botones, el teclado y un simulador de manos; sin generador, solo el catálogo.

### 3. Gestos

**Gestos** (la red reconoce 8 clases por mano: ninguno, palma abierta, pellizco, señalar, victoria, puño, pulgar arriba y pulgar abajo. Los gestos de dos manos los combina el visor):

| Gesto | Acción |
|---|---|
| Pellizco (pulgar + índice) | Tomar y mover el objeto |
| Palma abierta | Desplazarse por el espacio |
| Las dos palmas | Rotar siguiendo las manos, con inercia (si se sueltan girando, sigue y frena solo) |
| Pellizco con las dos manos | Separar para agrandar, juntar para achicar |
| Señalar (índice) | Cursor; apuntar 0,8 s a una pieza la elige |
| Pellizco sobre la pieza elegida | Sacar solo esa pieza y moverla (arrastrar y soltar) |
| Victoria (V) | Analizar la pieza elegida; sin pieza, abrir o cerrar el despiece |
| Puño | Salir del análisis o reiniciar la vista |
| Pulgar arriba / abajo | Confirmar un pedido / deshacer |

**Red neuronal de gestos (MLP, la que usa el visor por defecto):**
- Arquitectura: **63 → 128 → 64 → 8** (63 = 21 puntos × 3 coordenadas). **16.968 parámetros**. Pesa menos de 1 MB y corre en el navegador.
- Datos: **HaGRID**, dataset público de fotos de gestos (licencia CC BY-SA 4.0). Se le pasaron ~760 mil fotos por MediaPipe y quedaron **552.329 manos de unas 31 mil personas**. En el repositorio van solo las coordenadas, sin fotos, comprimidas en 5 archivos `.xz` (~90 MB).
- Partición **por persona** (nadie aparece en dos sets): 391.215 manos de entrenamiento, 75.703 de validación y 85.411 de prueba (4.687 personas que la red nunca vio).
- Resultado en prueba: **98,1 % de exactitud**, F1 macro 0,972, y confunde una mano sin gesto con un gesto solo el **1,2 %** de las veces. Las reglas escritas a mano dan **62,4 %** y 55 % de falsas alarmas.
- Se probaron 5 tamaños, hasta 724 mil parámetros: ninguno superó a la de 17 mil. Con aumento de datos, F1 de validación 0,962 contra 0,958 sin él.
- Sesgo medido: acierta 96,3 % con la mano derecha y 97,5 % con la izquierda.

**Árbol de decisión (alternativa legible):** usa 19 rasgos que se pueden leer (qué tan estirado está cada dedo, los ángulos de las articulaciones, las distancias entre puntas). Tiene profundidad 49 y 3.558 hojas, y da 95,5 % en prueba. El visor muestra el camino de preguntas que siguió. Comparación en prueba (42.875 manos):

| Clasificador | Exactitud | Falsas alarmas |
|---|---|---|
| Reglas a mano | 62,4 % | 55,3 % |
| Árbol de decisión (en el visor) | 95,5 % | 2,4 % |
| Bosque aleatorio (300 árboles) | 97,3 % | 1,6 % |
| Gradient boosting | 97,3 % | 1,2 % |
| Red neuronal | 97,9 % (98,1 % con el dataset ampliado) | 1,3 % |

Todas las diferencias son significativas por McNemar (p < 10⁻¹⁰⁰). Hay un árbol de profundidad 4 listo para mostrar en una diapositiva (`ml/resultados/arbol_reglas.txt`).

**Algoritmo genético (neuroevolución):** busca automáticamente la arquitectura de la red.
- Genoma: profundidad, ancho, conexiones residuales, LayerNorm, activación, dropout, tasa, weight decay, lote y aumento de datos.
- Operadores: torneo de 3, cruce uniforme, mutación 0,25, elitismo 2.
- Se comparó contra una búsqueda al azar con el mismo presupuesto (119 redes cada una; el genético tardó 69 min).
- Resultado honesto: el genético apenas le ganó al azar (F1 0,9596 contra 0,9581). Su mejor red (6×768, 3,2 M de parámetros) da 98,01 % contra 97,93 % de la red chica, con McNemar p = 0,13: **no es una diferencia real**. Por eso se quedó la red chica. Conclusión: la red de 2 capas ya está en el techo de lo que dan estos datos.

**Filtro de Kalman adaptativo:** 63 filtros (uno por coordenada) con un modelo de velocidad constante. Cuando la mano acelera, el filtro lo detecta (por la innovación normalizada, un estadístico chi-cuadrado) y suaviza menos. Con la mano quieta, el temblor baja un **46 %**. Con la mano en movimiento, el atraso queda en 4,9 ‰ de la imagen; sin la adaptación serían 56 ‰.

### 4. Pedidos en lenguaje natural

Todos los intérpretes devuelven el mismo JSON y pasan por los mismos filtros. Entienden 13 acciones: mostrar, rotar, escalar, vista, reiniciar, desmontar, explotar, armar, aislar, conjunto, usar_motor, consultar y desconocido.

**Cadena:** primero Claude, si hay clave y crédito (si falla, queda en pausa 10 minutos para no sumar ~1 s a cada pedido). Después, el **consenso de los modelos propios**. Al final, las reglas. Por voz se puede forzar un motor: "usá el transformer".

**Datos de entrenamiento:** **109.524 frases** generadas con plantillas rioplatenses: el 10 % con errores de tipeo y otras dichas por voces sintéticas y transcritas por Whisper, para que los modelos aprendan los errores de la voz. Los objetos salen de **9.828 nombres de WordNet en español** (artefactos, ropa, animales, comida, partes del cuerpo…), y el 10 % se reserva para medir con objetos no vistos. La evaluación usa sets escritos a mano que nunca se usaron para entrenar.

**Los cinco motores propios:**

| Motor | Qué es | Tamaño |
|---|---|---|
| Reglas | Expresiones y listas escritas a mano | — |
| Naive Bayes | Multinomial sobre palabras y pares de palabras (53.490 rasgos), con suavizado de Laplace y calibración por temperatura | 2,5 MB |
| Intérprete con embeddings | Vector de la frase con **multilingual-e5-small**, una regresión logística para la acción y un etiquetador (red de 512) que marca qué palabras son el objeto | 2,2 MB |
| **Transformer** (deep learning) | **XLM-RoBERTa large** (559,9 M de parámetros) ajustado entero con dos cabezas: la acción y el etiquetado **BIO** del objeto, decodificado con **Viterbi** | 2,1 GB |
| **LLM propio** | **Qwen2.5-1.5B-Instruct** ajustado con **LoRA** (r=16, 18,5 M de parámetros entrenables de 1.562 M, 1,2 %). Genera `accion|objeto` | 2,9 GB |

**Transformer, cómo se eligió:** se entrenaron dos candidatos con validación. BETO (BERT en español, 110 M) sacó un puntaje de 0,9859 y tarda 36 ms por frase. **XLM-R large sacó 0,9896, tarda 121 ms por frase y es el instalado** (02/11, 210 min de entrenamiento, 2 épocas). Con objetos no vistos acierta 98,2 % (acción + objeto exacto) y con frases generadas reservadas, 99,3 %.

**LLM propio 1.5B:** 162 min de entrenamiento en CPU y 2,7 s por frase. En validación acierta 98,2 % con objetos no vistos y 100 % de los comandos. El 0.5B (también entrenado, 0,95 GB) da 96,4 % y 91,7 %, y queda de respaldo.

**Consenso (el motor por defecto sin Claude):**
- Es un **producto de expertos log-lineal**: log P(acción) = Σ wᵢ · log pᵢ(acción) + b. Combina las distribuciones de los cinco motores.
- Los pesos se ajustan por **máxima verosimilitud con L-BFGS**. Se compararon contra un algoritmo genético y contra el voto simple, con validación cruzada de 5 partes sobre 138 frases.
- Pesos resultantes: transformer 0,47, LLM 0,31, Bayes 0,19, embeddings y reglas 0 (no suman nada que el transformer no tenga).
- **Decisión bayesiana**: si la probabilidad de equivocarse (1 − P) supera el costo de preguntar (0,2), pide confirmación en lugar de ejecutar.
- Ventaja principal: está mejor calibrado (ECE 0,008 contra 0,162 del promedio), así que su confianza significa algo.

**Resultados de punta a punta** (medidos el 30/10, **con BETO como transformer**, antes de instalar XLM-R):

| Set (frases) | Reglas | Bayes | Embeddings | LLM 1.5B | Transformer | **Consenso** |
|---|---|---|---|---|---|---|
| Prueba (38) | 84 % | 87 % | 87 % | 97 % | 97 % | **97 %** |
| Validación (59) | 66 % | 81 % | 95 % | 98 % | 100 % | **100 %** |
| Frases nuevas (50) | 70 % | 72 % | 92 % | 98 % | 92 % | **100 %** |
| Objetos no vistos, prueba (45) | 60 % | 56 % | 89 % | 98 % | 98 % | **98 %** |
| Comandos, prueba (38) | 79 % | 92 % | 89 % | 95 % | 97 % | **97 %** |
| Latencia mediana | 0 ms | 0 ms | 15 ms | 1,5–2,5 s | 33 ms | 4,3–5,7 s |

El consenso acierta **305 de 309 frases (98,7 %)** en los siete sets. El transformer solo acierta 300 de 309, pero responde casi al instante. Con sets de 38 a 59 frases, las diferencias entre los cuatro mejores **no son estadísticamente significativas** (McNemar p ≥ 0,125); contra las reglas y Bayes, sí. **Claude no se midió** porque no había crédito en la API.

**Corrección de errores de tipeo:** distancia de Damerau-Levenshtein con un **BK-tree** sobre ~9.700 palabras (revisa ~25 % del vocabulario en lugar de todo) y un modelo de **canal ruidoso** con frecuencias reales del español (wordfreq). Por ejemplo, "valbula de asero" se corrige a "válvula de acero", pero "compresor" no se toca porque es una palabra común.

### 5. Voz

- **Whisper large-v3** corre local en el procesador (faster-whisper, int8, licencia MIT). El audio no sale de la computadora. La transcripción usa **beam search** con 5 hipótesis. Sin backend, se usa el reconocimiento del navegador.
- Detector de voz propio: umbral adaptativo (media del ruido más 3,5 desvíos, que se actualiza solo en silencio).
- Modos: pulsar para hablar (V), **manos libres con la palabra de activación "holograma"** (M) y respuestas habladas (J).
- Comandos sobre los modelos: "usá el árbol para los gestos", "¿qué modelo respondió?", "¿qué tan seguro estás?", "¿por qué hiciste eso?".
- Evaluación con 126 frases, 3 voces sintéticas (Argentina, España y México) y ruido de fondo:
  - Error por palabra (WER): de 6 a 21 %.
  - Acción correcta: entre 82 y 90 % por voz, contra ~98 % escrita.
  - La palabra "holograma" se detectó el 100 % de las veces, sin activaciones falsas.
  - Whisper large-v3 tarda unos **8 s por frase** en CPU. Limitación: son voces sintéticas; en un aula con personas reales va a dar peor.

### 6. Catálogo 3D y búsqueda

- **Modelos propios:** 5 piezas industriales (engranaje, motor eléctrico, rodamiento, tornillo, válvula) más motores de combustión generados por código (en línea, V, bóxer, de 1 a 12 cilindros). Son 12 MB.
- **Objaverse (Sketchfab):** la selección tiene **17.081 modelos en 1.152 categorías** (131 de máquinas y piezas escritas a mano, más las categorías LVIS), y suma **136 GB**. El tope configurado es 250 GB. Hay **12.201 modelos descargados (~108 GB)**. Solo se aceptan licencias **CC BY, CC BY-SA o CC0**, y el visor muestra el autor y la licencia de cada modelo.
- **Filtro de coherencia:** la descripción del modelo tiene que parecerse a su categoría (coseno ≥ 0,45 con embeddings multilingües). Sin este filtro, "engranaje" traía cascos ("gear") y "válvula" traía modelos de la empresa Valve.
- **Búsqueda híbrida:** primero por palabras clave y sinónimos, después por embeddings (MiniLM multilingüe) con un umbral de 0,62. Acierta **90 %**, contra 80 % solo con palabras y 75 % solo con embeddings. Se agregó una regla: una consulta de una sola palabra que no coincide con nada no usa embeddings. Antes, "silla" devolvía un tornillo.
- **Latencia del backend** (sin LLM): interpretar 1,7 ms + buscar 1,1 ms + bajar el modelo 11,7 ms = **14 ms** en total. La latencia de punta a punta en el navegador (hasta dibujar el modelo) **está pendiente de medir**.

### 7. Desarme y reconocimiento de piezas

- **Cualquier modelo se puede desarmar**, aunque su autor no lo haya hecho por piezas. Si el archivo tiene menos de 4 piezas, el visor separa las **partes sueltas** (grupos de triángulos que no se tocan). Si con eso no alcanza, o si una parte ocupa más del 80 % del modelo, lo corta en **secciones por k-medias**. Medido sobre 300 modelos: todos quedan con entre 4 y 80 piezas (mediana 7), en 44 ms de mediana.
- **A\*** planifica el orden de desarme sin que las piezas choquen. El estado es qué piezas ya salieron; una pieza sale si la caja que barre no choca con las que quedan. La heurística es admisible, así que el plan es óptimo.
- Vista explotada, desmontar y montar de a una pieza, analizar una pieza sola (qué es, el material, cuánto mide) y deshacer.
- **Reconocimiento de piezas de motores:** una **PointNet** (red que procesa nubes de puntos, 256 puntos por pieza) le pone nombre a cada pieza: pistón, biela, cigüeñal, cárter, rotor… Tiene **45 clases**.
  - Se entrenó con **1.200 motores generados por código** (85.178 piezas), con deformaciones, piezas faltantes y piezas ajenas al motor.
  - Con familias de motores que nunca vio acierta **98,1 %**. Corre en el navegador en 150 a 550 ms y da lo mismo que en PyTorch.
  - Limitación: **no está medido con modelos reales** de Objaverse (no hay motores etiquetados). Por eso solo nombra una pieza si tiene al menos 80 % de confianza, y avisa que el nombre es una estimación.

### 8. Generación 3D (cuando el objeto no está en el catálogo)

- **Por piezas con el LLM** (por defecto con clave de API): Claude diseña el objeto como un ensamble de formas simples (cilindros, conos, engranajes, cajas…) con medidas, material, orden de desmontaje y dirección de salida. El backend valida el diseño y arma un `.glb` con una pieza por nodo, así que lo generado también se desarma. No necesita GPU.
- **TripoSR o Shap-E en Colab con GPU:** da formas más orgánicas, pero en una sola malla, así que no se desarma.
- Lo generado se guarda en el catálogo y se marca como "generado por IA, no apto para fabricación sin revisión".

### 9. Interfaz y transparencia

- Modo holograma (tecla H), catálogo en pestañas (C), ajustes de sensibilidad y umbral (K), **modo presentación** (P), paneles que se arrastran y se guardan como "disposiciones".
- **`red.html`**: la mano entra a la red en vivo y se ven las activaciones, las conexiones que más pesan, la salida softmax contra el umbral, y la misma mano en el árbol (con su camino) y en las reglas. Muestra también la curva del genético.
- **`llm.html`**: el recorrido de cada pedido. Muestra qué dijo cada motor con su distribución, el consenso con su fórmula, la decisión de preguntar o ejecutar y los filtros. Tiene un laboratorio para probar frases.
- **Actividad IA** (L): un panel en vivo que se puede abrir en otra pantalla.

### 10. Seguridad y uso responsable

- Validación de toda respuesta de un modelo: esquema JSON, objetos fuera de dominio, límites de escala y pedidos por minuto (`safety.py`, `schemas.py`). Los `.glb` generados también se validan (cabecera, largo, tamaño máximo de 20 MB).
- **Privacidad:** el video y el audio se procesan en la computadora. Los únicos datos que salen son el texto del pedido a Claude (opcional) y la voz del navegador, solo si el backend está apagado.
- La evaluación y el entrenamiento nunca usan Claude, para no gastar crédito.
- Licencias abiertas en todos los datos (HaGRID CC BY-SA, WordNet CC BY, Objaverse CC BY/BY-SA/CC0) y atribución en el visor.
- Sesgos medidos: mano derecha contra izquierda en gestos, acentos en voz.

### 11. Entrenamiento

- Un solo comando entrena todo en orden: `python scripts/entrenar.py`. Se puede **pausar con la tecla P** y retomar con `--continuar`. Los puntos de control guardan los pesos, el optimizador, el plan de la tasa, el orden de las frases y el estado del azar, así que al retomar sigue igual que si no se hubiera cortado.
- Todo se entrenó en **CPU, sin GPU**.

| Paso | Tiempo aprox. en CPU |
|---|---|
| Naive Bayes | 1 min |
| Intérprete con embeddings | 25 min |
| Transformer XLM-R large | 3,5 h (estimado hasta 16 h) |
| LLM propio 1.5B | 2,7 h |
| **LLM propio 7B** | **~157 h en total** |
| Consenso | 15 min |
| Red de gestos (5 tamaños) | 10 min |
| Árboles | 5 min |
| Reconocimiento de piezas (PointNet) | 12–20 min |
| Algoritmo genético | 69 min |

### 12. Cuánto pesa cada cosa

| Qué | Tamaño |
|---|---|
| Red de gestos + árbol + PointNet de piezas + demo (lo que corre en el navegador) | **0,8 MB** en total |
| Librerías del frontend (Three.js, MediaPipe y su modelo de manos) | 27 MB |
| Modelos 3D propios y motores generados | 12 MB |
| Naive Bayes / intérprete con embeddings / consenso | 2,5 MB / 2,2 MB / < 0,1 MB |
| Transformer XLM-R large ajustado | 2,1 GB |
| LLM propio Qwen 1.5B ajustado | 2,9 GB |
| LLM propio Qwen 0.5B ajustado | 0,95 GB |
| Punto de control del LLM 7B (en entrenamiento) | 1,8 GB |
| Modelos base descargados de Hugging Face (Qwen 0.5B/1.5B/7B, XLM-R, BETO, e5, MiniLM, Whisper, voces Piper) | **~93 GB** |
| Landmarks de HaGRID (552 mil manos, comprimidos) | ~90 MB |
| Frases de entrenamiento e intérprete | ~8,5 MB |
| Catálogo Objaverse descargado (12.201 modelos) | **~108 GB** (selección completa: 136 GB) |
| Metadatos de Objaverse (descripciones Cap3D, LVIS) | 0,8 GB |
| Repositorio git | 163 MB |

**Código:** unas **19 mil líneas** propias: ~7.300 de JavaScript en el frontend, ~4.800 de Python en el backend, ~4.900 de Python de entrenamiento, ~1.100 de evaluación y ~900 de scripts. Hay pruebas automáticas en Node (gestos, piezas, MLP JS = PyTorch, voz, holograma, reconocimiento de piezas) y en pytest (backend y normalización). El script `scripts/check_all.sh` verifica todo antes de la defensa.

### 13. Técnicas de la materia que cubre

Redes neuronales (MLP, PointNet), deep learning (transformer ajustado), LLM (Claude y uno propio con LoRA), algoritmos genéticos, árboles de decisión y ensambles, Naive Bayes, estadística (bootstrap, intervalos del 95 %, McNemar, calibración con ECE), filtro de Kalman, búsqueda (A*, Viterbi, beam search, BK-tree), optimización (L-BFGS), teoría de la decisión, embeddings y búsqueda semántica, visión por computadora, reconocimiento de voz e IA generativa 3D.

Varias de estas técnicas **no estaban en la propuesta original**: se sumaron a pedido del equipo, y cada una cumple una función real y se mide contra lo que ya había.

### 14. Estado actual y advertencias (respetalas)

- **LLM propio 7B (Qwen2.5-7B, LoRA r=64, 161 M de parámetros entrenables, 2,1 %)**: está **en entrenamiento y en pausa, al 81 %** (paso 901 de 1.116, ~126 h de CPU). Le faltan ~30 h, así que **no llega a la defensa**. La demo usa el **Qwen 1.5B**. Presentarlo como trabajo en curso.
- El consenso y la tabla de punta a punta se midieron con BETO y el LLM 1.5B. Falta volver a medirlos con XLM-R (instalado el 02/11) y, más adelante, con el 7B.
- Parte de la documentación vieja dice que el transformer es BETO y que la red de gestos da 97,9 %. Los datos vigentes son XLM-R large y 98,1 % con el dataset ampliado.
- No medidos: Claude (sin crédito), la latencia de punta a punta en el navegador, el reconocimiento de piezas con modelos reales, la voz con personas reales y la precisión de Whisper turbo.
- Los tiempos del problema (cuánto tarda hoy un equipo comercial sin el producto) son **estimaciones**, no mediciones.

---

## Formato de salida

1. **Informe técnico** con secciones numeradas y tablas.
2. **Guion de diapositivas**: un bloque por diapositiva, con *Título · Viñetas · Visual · Notas del orador · Tiempo*.
3. **Preguntas del jurado** con sus respuestas.
4. **Glosario**.

Si podés generar archivos, generá además el **.pptx** con un diseño oscuro y acentos cian/neón (como el holograma), tablas legibles y las notas del orador en cada diapositiva.
