# Técnicas de IA del proyecto

Cada técnica cumple una función real en el producto y se mide contra lo que ya había, sobre los mismos sets de validación y prueba. Las diferencias se informan con intervalos de confianza del 95 % (bootstrap) y, entre dos modelos, con el test de McNemar (`eval/estadistica.py`). Las decisiones están en [decisiones.md](decisiones.md); los números completos, en [metricas.md](metricas.md).

> **Desvío de la propuesta.** Los algoritmos genéticos, los árboles de decisión, Naive Bayes, el filtro de Kalman, A*, el transformer ajustado, el LLM propio y el consenso no estaban en la propuesta: se sumaron a pedido del equipo para cubrir más técnicas de la materia. La voz local sí estaba (sección 16.1: "preferir transcripción local"), aunque la propuesta la ponía primera en la lista de recortes.

## Mapa

| Técnica | Para qué se usa | Dónde |
|---|---|---|
| Redes neuronales (MLP) | Reconocer el gesto de la mano | `ml/entrenar_red_gestos.py` → `frontend/js/mlp.js` |
| Algoritmo genético (neuroevolución) | Diseñar la arquitectura de esa red | `ml/evolucion/neuroevolucion.py` |
| Árboles de decisión (árbol, bosque, boosting) | Reconocer el gesto con reglas aprendidas y legibles | `ml/arboles/` → `frontend/js/arbol.js` |
| Deep learning: transformer ajustado (BETO) | Interpretar pedidos: acción + objeto (BIO) | `ml/intencion_transformer/` → `backend/intent_transformer.py` |
| LLM propio (Qwen2.5 + LoRA) | Interpretar pedidos generando "accion\|objeto" | `ml/llm_propio/` → `backend/llm_propio.py` |
| LLM de Claude | Interpretar pedidos (si hay crédito) | `backend/intent.py` |
| Embeddings + clasificadores | Interpretar pedidos (intérprete v2) y buscar en el catálogo | `backend/intent_modelo.py`, `backend/retrieval.py` |
| Naive Bayes (estadística) | Interpretar pedidos; línea base clásica | `ml/intencion/entrenar_bayes.py` → `backend/intent_bayes.py` |
| Consenso (optimización + teoría de la decisión) | Elegir la mejor respuesta entre todos los motores | `ml/consenso/ajustar_consenso.py` → `backend/consenso.py` |
| Calibración (escalado de temperatura, ECE) | Que la confianza signifique algo | `eval/estadistica.py` |
| Bootstrap y McNemar | Saber si una diferencia es real | `eval/estadistica.py`, `eval/eval_intent.py` |
| Filtro de Kalman adaptativo | Sacar el temblor de los puntos de la mano | `frontend/js/kalman.js` |
| Búsqueda: beam search | Whisper elige la transcripción más probable | `backend/voz.py` |
| Búsqueda: Viterbi (programación dinámica) | La secuencia BIO válida más probable | `backend/intent_transformer.py` |
| Búsqueda: Levenshtein + BK-tree + canal ruidoso | Corregir errores de tipeo y de transcripción | `backend/busqueda_difusa.py` |
| Búsqueda: A* | Orden y dirección para desarmar un modelo sin que las piezas choquen | `frontend/js/piezas.js` |
| Reconocimiento de voz (Whisper large-v3) | Pedidos y comandos por voz, sin mandar el audio afuera | `backend/voz.py`, `frontend/js/voz.js` |
| Detector de voz con umbral adaptativo | Saber cuándo empieza y termina una frase | `frontend/js/voz.js` |

## Gestos

**Red neuronal.** Perceptrón multicapa 63 → 128 → 64 → 8, entrenado con PyTorch y aumento de datos sobre 196 mil manos de HaGRID. En prueba (42.875 manos de 3.416 personas que no vio) da **97,9 % de exactitud** (reglas: 62,4 %).

**Neuroevolución.** Un algoritmo genético propio busca la arquitectura:

- **Genoma:** profundidad, ancho, residual, LayerNorm, activación ReLU/GELU/SiLU, dropout, tasa, weight decay, lote e intensidad del aumento.
- **Operadores:** torneo de 3, cruce uniforme, mutación 0,25 y elitismo 2, con caché de aptitudes.
- **Aptitud:** F1 en validación tras un entrenamiento corto.
- **Comparación justa:** una búsqueda al azar con el mismo presupuesto de redes entrenadas.
- **Final:** los 3 mejores se entrenan completos y se compara contra la v2 con McNemar.

Si gana en validación, se exporta como versión 3: `mlp.js` lee residual, LayerNorm y GELU/SiLU, y su salida coincide con PyTorch a 10⁻⁷.

**Resultado** (119 redes por método, 69 min el genético):

- **Búsqueda:** el genético encontró un F1 de 0,9596 en validación corta; la búsqueda al azar, 0,9581. Con este presupuesto, el genético apenas le gana al azar.
- **Finalistas, entrenados completos:** 5×512 ReLU con LayerNorm (F1 0,9571), la misma con residual (0,9562) y 6×768 GELU residual con LayerNorm (0,9575, 3,2 M de parámetros).
- **En prueba:** la mejor da 98,01 % [97,88–98,15] contra 97,93 % [97,79–98,06] de la v2. McNemar: 298 manos que acierta solo la nueva contra 261 solo la v2, p = 0,13. No es una diferencia real.
- **Decisión:** no se exporta, porque no le gana a la v2 en validación. La red de 2 capas (128 y 64) ya está en el techo de lo que dan estos datos; más profundidad no ayuda. La curva de la evolución está en `red.html` y en `frontend/ml/neuroevolucion.json`.

**Árboles de decisión.** Los rasgos son legibles (19): qué tan estirado está cada dedo, el ángulo de cada articulación y las distancias entre puntas. Son los mismos en Python y en JS, y una prueba lo controla. Resultados en prueba:

| Clasificador | Exactitud | F1 macro | Falsas alarmas en "ninguno" |
|---|---|---|---|
| Reglas escritas a mano | 62,4 % | 0,695 | 55,3 % |
| **Árbol de decisión** (profundidad 49, 3.558 hojas; en el visor) | **95,5 %** | 0,937 | 2,4 % |
| Bosque aleatorio (300 árboles) | 97,3 % | 0,961 | 1,6 % |
| Gradient boosting | 97,3 % | 0,963 | 1,2 % |
| Bosque sobre las 63 coordenadas crudas | 97,9 % | 0,969 | 1,1 % |
| Red neuronal v2 | 97,9 % | 0,969 | 1,3 % |

- **Árbol frente a reglas:** el árbol aprende de los datos los umbrales que las reglas ponían a ojo y les gana por 33 puntos.
- **Ensambles:** los ensambles de árboles se acercan a la red.
- **Diferencias significativas:** todas son significativas por McNemar (p < 10⁻¹⁰⁰).
- **Rasgos más importantes:** según el bosque, hacia dónde apunta el pulgar, el medio, el anular, el índice y la curva del índice.
- **Árbol legible:** un árbol de profundidad 4 está en `ml/resultados/arbol_reglas.txt` para leerlo en una diapositiva.

**Filtro de Kalman adaptativo.** Usa un modelo de velocidad constante por coordenada. La intensidad de la aceleración sube cuando la innovación normalizada promedio de los 63 filtros (un estadístico chi-cuadrado) indica que la mano aceleró. Los parámetros se eligen con una grilla sobre el simulador:

- **Mano quieta:** el temblor baja de 2,77 ‰ a 1,50 ‰ de la imagen (−46 %).
- **Mano en movimiento:** el atraso queda en 4,9 ‰. Sin adaptación serían 56 ‰.

## Pedidos (lenguaje)

Todos los motores devuelven el mismo JSON y pasan por los mismos filtros de seguridad. Exactitud de punta a punta:

| Set (frases) | Reglas | Naive Bayes | Embeddings (v2) | LLM propio 1.5B | Transformer BETO | **Consenso** |
|---|---|---|---|---|---|---|
| Prueba (38) | 84 % | 87 % | 87 % | 97 % | 97 % | **97 %** |
| Validación (59) | 66 % | 81 % | 95 % | 98 % | 100 % | **100 %** |
| Nuevas (50) | 70 % | 72 % | 92 % | 98 % | 92 % | **100 %** |
| Objetos, validación (55, objeto exacto) | 56 % | 53 % | 95 % | 98 % | 96 % | **98 %** |
| Objetos, prueba (45, objeto exacto) | 60 % | 56 % | 89 % | 98 % | 98 % | **98 %** |
| Comandos, validación (24) | 92 % | 100 % | 92 % | 100 % | 100 % | **100 %** |
| Comandos, prueba (38) | 79 % | 92 % | 89 % | 95 % | 97 % | **97 %** |
| Latencia mediana | 0 ms | 0 ms | 15 ms | 1,5–2,5 s | 33 ms | 4,3–5,7 s |

Todos los motores ya están entrenados con los comandos. Intervalos y McNemar por set: `eval/eval_intent.py --motor todos`. Con 38 a 59 frases por set, las diferencias entre los cuatro mejores no son significativas (McNemar p ≥ 0,125). Contra las reglas y Naive Bayes, sí lo son en validación (p ≤ 0,021). Claude no se midió: no hay crédito.

El consenso es el que menos falla: en total, 305 de 309 frases (98,7 %). Lo que cuesta es la latencia: corre los cinco motores a la vez en el procesador, y el LLM grande se lleva casi todo el tiempo. Para respuestas instantáneas, «usá el transformer» da casi lo mismo (300 de 309) en 33 ms.

- **Transformer (deep learning).** Es BETO (110 M de parámetros, BERT en español), ajustado entero con dos cabezas: la acción y el etiquetado BIO del objeto. La secuencia BIO se decodifica con Viterbi. Se probó también XLM-RoBERTa large (560 M, 1 época por el tiempo en CPU): puntaje de validación 0,964 contra 0,986 de BETO, 127 ms por frase. Se quedó BETO. El RoBERTa large en español del Plan TL ya no publica sus pesos.
- **LLM propio.** Es Qwen2.5-Instruct ajustado con LoRA (1,7 % de los parámetros), que escribe "accion|objeto". Para el consenso calcula también P(acción) de las 13 acciones a la fuerza, en una pasada. Hay dos tamaños, entrenados con los comandos. En validación:

| | Acción | Objetos no vistos (acción + objeto exacto) | Comandos | Por frase |
|---|---|---|---|---|
| Qwen2.5 0.5B | 96,6 % | 96,4 % | 91,7 % | 0,8 s |
| **Qwen2.5 1.5B** (el que se usa) | 96,6 % | **98,2 %** | **100 %** | 2,7 s |

El 1.5B le gana en objetos y comandos, y es el que queda por defecto; con `LLM_PROPIO=chico` se usa el otro. La P(acción) forzada que entra al consenso acierta sola 82,6 %, menos que generando. Por eso el consenso le da peso, pero no todo.
- **Naive Bayes.** Multinomial sobre palabras y pares de palabras, con el suavizado de Laplace elegido en validación. Es la línea base estadística: con frases parecidas a las de entrenamiento rinde como los modelos (87 % en prueba); con objetos de todo tipo, no (56 %). La calibración con temperatura (T = 3,9) baja el ECE de 0,091 a 0,046.
- **Consenso.** Combina a todos con un producto de expertos ponderado: log P(a) = Σ w·log p(a) + b. Los pesos se ajustan por máxima verosimilitud con L-BFGS y se comparan contra un algoritmo genético y contra el voto simple, con validación cruzada.
  - **Decisión bayesiana:** se pide confirmación si (1 − P) supera el costo de preguntar (0,2).
  - **Ajuste final** (138 frases de validación, validación cruzada de 5 partes):

    | Combinación | Exactitud | ECE |
    |---|---|---|
    | Mejor motor solo (transformer) | 98,6 % [96–100] | 0,009 |
    | Voto simple | 97,8 % [95–100] | — |
    | Promedio de probabilidades | 97,8 % [95–100] | 0,162 |
    | Log-lineal, L-BFGS | 98,6 % [96–100] | 0,008 |
    | Log-lineal, algoritmo genético | 98,6 % [96–100] | 0,005 |

  - **Pesos (L-BFGS):** transformer 0,47, LLM propio 0,31, Bayes 0,19; embeddings y reglas 0. Los dos últimos no suman nada que el transformer no tenga.
  - **Qué se gana:** en validación empata con el transformer, pero en la prueba de frases nuevas lo corrige (100 % contra 92 %). La combinación log-lineal está mejor calibrada que el voto o el promedio, y eso es lo que usa la decisión de preguntar.
- **Cadena.** Primero Claude, si hay crédito; si falla por falta de crédito o de clave, queda en pausa 10 minutos para no sumar ~1 s a cada pedido. Después, el consenso y al final las reglas. Por voz se puede elegir un motor: «usá Naive Bayes».

## Búsqueda

- **Corrección difusa.** Usa la distancia de Damerau-Levenshtein (programación dinámica) y un BK-tree sobre unas 9.700 palabras: para cada palabra mal escrita revisa ~25 % del vocabulario, en lugar de todo. Una palabra con frecuencia en español (wordfreq, Zipf ≥ 2) nunca se corrige. Entre las candidatas gana la de mayor log P(palabra) − 2·errores (modelo de canal ruidoso). Ejemplos: "rulemam" → rulemán (catálogo), "valbula de asero" → válvula de acero, "compresor" no se toca. Se aplica cuando la búsqueda por palabras del catálogo no encuentra nada.
- **A\* para el desarme.** El estado es qué piezas ya salieron. Una pieza sale en una dirección si la caja que barre no choca con ninguna de las que quedan y sobresalen. Los costos son 1 por pieza, 0,5 por cada giro y el recorrido. La heurística (las piezas que faltan) es admisible, así que el plan es óptimo. Se usa en los modelos generados sin orden. Si las piezas están trabadas, el plan igual termina y la pieza queda marcada.
- **Viterbi** (BIO del transformer) y **beam search** (Whisper, 5 hipótesis).

## Voz

- **Reconocimiento local.** Whisper large-v3 corre en el procesador (int8, faster-whisper) con un texto inicial con el vocabulario del visor. El audio no sale del equipo.
- **Detector de voz.** Es un umbral adaptativo: la media del ruido más 3,5 desvíos, actualizada solo en silencio.
- **Modos.**
  - **Pulsar para hablar:** micrófono o tecla V.
  - **Manos libres:** tecla M; atiende lo que empieza con «holograma», y con «holograma» solo responde «¿sí?».
  - **Respuestas habladas:** tecla J para apagarlas o prenderlas.
- **Comandos de los modelos:** «usá el árbol para los gestos», «pasá al transformer», «¿qué modelo respondió?», «¿qué tan seguro estás?», «¿por qué?».
- **Evaluación con voces Piper** (`eval/eval_voz.py`). Se usaron las 126 frases de prueba, nuevas y comandos, con 3 voces sintéticas y ruido rosa; los pedidos los interpretó el consenso:

  | Voz | Ruido | WER | Acción como con la frase escrita | Acción correcta |
  |---|---|---|---|---|
  | Argentina (daniela) | limpio / 20 dB / 10 dB | 16,4 / 15,6 / 18,4 % | 88 / 90 / 87 % | 86 / 89 / 87 % |
  | España (davefx) | limpio / 20 dB / 10 dB | 7,3 / 6,9 / 6,4 % | 92 / 92 / 92 % | 90 / 90 / 90 % |
  | México (ald) | limpio / 20 dB / 10 dB | 17,3 / 16,9 / 21,5 % | 89 / 92 / 84 % | 87 / 90 / 82 % |

  - **Palabra de activación:** «holograma» se detectó en el 100 % de las frases que la tenían, sin ninguna activación falsa en las demás.
  - **Qué pasa con el ruido:** el ruido a 20 dB casi no cambia nada. A 10 dB, la voz mexicana pierde unos 5 puntos.
  - **Cuánto se pierde:** escrita, la acción sale bien en ~98 % de estas frases; por voz, entre 82 y 90 %. Un WER del 16 % no significa perder el 16 % de los pedidos: muchas veces el error está en una palabra que no cambia la acción.
  - **Latencia:** Whisper large-v3 tarda unos 8 s por frase en este procesador. Es el modelo más preciso, pero lento para usar en vivo. `WHISPER_MODELO=large-v3-turbo` (o `medium`) es varias veces más rápido; no se midió su precisión.
  - **Limitación:** son voces sintéticas y limpias; con personas en un aula va a dar peor.

## Vistas para entender los modelos

- **`red.html` (red neuronal en vivo).** Muestra la mano que ve el visor pasando por la red:
  - capas, activaciones y las conexiones que más aportan;
  - salida softmax con el umbral y probabilidades en el tiempo;
  - la misma mano en la red, el árbol (con su camino) y las reglas, comparados;
  - arquitectura, cómo funciona paso a paso y la curva del algoritmo genético.

  Sin visor abierto, recorre manos reales de HaGRID que la red no vio.
- **`llm.html` (LLM en vivo).** Muestra el recorrido de cada pedido:
  - entrada (texto o voz con Whisper);
  - cada motor con su distribución y lo que cuenta: la salida cruda del LLM, las etiquetas BIO, las palabras que pesaron en Bayes y la probabilidad de cada palabra de ser el objeto;
  - el consenso con su fórmula y la decisión bayesiana;
  - la validación y la ejecución.

  Tiene un laboratorio para probar frases con cualquier motor, sin tocar el visor.
