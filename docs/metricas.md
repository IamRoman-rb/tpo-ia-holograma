# Resultados de la evaluación

Ver el plan completo en la sección 15 de [propuesta.md](propuesta.md). Para reproducir los números (desde la raíz):

```bash
python eval/eval_intent.py                                  # intención: reglas, modelo entrenado y LLM (el LLM necesita ANTHROPIC_API_KEY)
python eval/eval_busqueda.py                                # búsqueda: comparación de métodos y barrido de umbral
python eval/eval_busqueda.py --set validacion --solo-producto   # búsqueda en el set que no se usó para calibrar
python eval/medir_latencias.py                              # latencias del backend (con LIMITE_PEDIDOS_MINUTO=1000)
python eval/pruebas_usuarios/resumir.py                     # totales de las pruebas con usuarios
```

Los detalles por frase y por pedido quedan en `eval/resultados/`.

| Área | Métrica | Meta | Resultado | Notas |
|---|---|---|---|---|
| Gestos | Precisión por gesto (persona no vista) | ≥ 90 % | **Red v2: 96,3 %** (recall medio de los 7 gestos, fotos de HaGRID) · con cámara: pendiente | Ver "Gestos con datos de internet" abajo |
| Gestos | Reglas vs. red | Reportar | **Exactitud 62,4 % vs. 97,9 %** · falsas alarmas 55,3 % vs. 1,3 % | 42.875 manos de 3.416 personas que la red no vio |
| Gestos | Falsos positivos en 60 s en reposo | ≤ 2 | Pendiente | Incluido en el protocolo de pruebas; necesita cámara |
| Latencia | Gesto → pantalla | < 100 ms | Pendiente | Con cámara: panel de depuración (tecla D), línea `fotograma` |
| Latencia | Pedido → modelo (catálogo) | < 2 s | **Backend: 14 ms** (mediana; p95 15 ms) · de punta a punta: pendiente | Medido con el **intérprete por reglas**. Con el LLM, la interpretación va a tardar del orden de segundos: hay que medirlo |
| Latencia | Pedido → modelo (generado) | Reportar | Pendiente | El notebook de Colab mide TripoSR y Shap-E |
| Generación | Modelos por piezas válidos, partes clave y calidad (10 pedidos) | Reportar | Pendiente | `eval/eval_ensamble.py`; necesita clave de API. Mide también segundos y costo por modelo |
| LLM | Aciertos de intención (38 frases) | ≥ 85 % | **Modelo entrenado: 87 % · reglas: 84 %** · LLM: pendiente | Con 45 pedidos de objetos de todo tipo que no vio, **modelo 91 %, reglas 60 %** (objeto exacto). Ver "Intérprete de pedidos entrenado" abajo. El LLM se evalúa cuando haya una clave de API |
| Búsqueda | Recuperación correcta (20 pedidos) | ≥ 80 % | **Híbrido: 90 % en calibración y 90 % en validación** | Ver el detalle abajo |
| Usabilidad | Éxito de tarea sin ayuda | ≥ 80 % | Pendiente | Protocolo con 5 tareas en `eval/pruebas_usuarios/protocolo.md` |

## Gestos con datos de internet (HaGRID)

**Versión 2 (la del visor):** `ml/extraer_hagrid.py` y `ml/entrenar_red_gestos.py` (PyTorch). Fuente: muestra de **254.661 fotos** de HaGRID (Hugging Face `cj-mills/hagrid-sample-250k-384p`, 6,8 GB, CC BY-SA 4.0). MediaPipe encontró y asignó **276.237 manos de 23.205 personas**. Partición fija por persona: 195.699 manos para entrenar, 37.663 para validar y **42.875 para la prueba (3.416 personas que la red no vio)**. La red v1 (entrenada con la muestra de 30k) se evaluó en la misma prueba: la partición por persona es la misma, así que tampoco vio a esas personas.

| Método | Exactitud | F1 macro | Gestos reconocidos (recall medio) | Falsas alarmas (manos sin gesto tomadas como gesto) |
|---|---|---|---|---|
| Reglas (`gestures.js`) | 62,4 % | 0,695 | 92,6 % | **55,3 %** |
| Red v1 (128-64, 34.503 manos, sin aumento) | 97,6 % | 0,966 | 95,4 % | 1,2 % |
| **Red v2** (128-64, 276.237 manos, con aumento, umbral 0,8) | **97,9 %** | **0,969** | **96,3 %** | **1,3 %** |

**Por clase (prueba, red v2):** palma 98 %, pellizco 97 %, pulgar abajo 97 %, pulgar arriba 97 %, puño 98 %, señalar 94 %, victoria 93 %, ninguno 99 %. Matrices de confusión: `ml/resultados/hagrid_{red_v2,reglas}_confusion.csv`; reporte completo: `ml/resultados/hagrid_red_v2_reporte.json`.

**¿Sirvió agrandar la red?** No. Con las 276.237 manos se compararon cinco tamaños (F1 macro en validación):

| Capas ocultas | Parámetros | F1 validación |
|---|---|---|
| **128-64** | **16.968** | **0,960** |
| 256-128 | 50.312 | 0,958 |
| 512-256 | 166.152 | 0,958 |
| 512-256-128 | 198.024 | 0,958 |
| 1024-512-256 | 723.720 | 0,960 |

Con 40 veces más parámetros, la red da lo mismo, y las grandes dejan de mejorar a las pocas épocas. El límite está en los datos y no en la red: 21 puntos con ruido de detección, gestos que se parecen ("señalar" y "victoria" son los de menor recall) y un pellizco que en HaGRID es el gesto "ok". Se queda la más chica: pesa 88 KB y en el navegador tarda microsegundos. Lo que sí mejoró fue tener **8 veces más datos** y el **aumento de datos** (manos espejadas, inclinadas ±25°, giradas en profundidad y con ruido): 0,960 de F1 en validación con aumento contra 0,954 sin aumento.

**Qué muestra la comparación con las reglas.** Las reglas encuentran casi todos los gestos, pero también "ven" gestos donde no los hay: una mano con cuatro dedos extendidos es una palma para ellas, y dos dedos juntos hacia arriba, una victoria. En el visor, cada falsa alarma es una acción que nadie pidió. La red aprendió esas diferencias de miles de ejemplos.

**Sesgo por mano:** la red v2 reconoce los gestos con la mano derecha (96,0 %) y con la izquierda (97,0 %) de forma parecida. **Por persona:** de 790 personas con al menos 3 gestos en la prueba, la mediana es 100 % y 27 (3,4 %) quedan por debajo del 80 %. Con la v1 eran 8 de 177 (4,5 %), pero en una prueba más chica.

**Versión 1 (para reproducirla):** `ml/entrenar_mlp.py` (scikit-learn) con la muestra de 31.833 fotos (34.503 manos de 7.798 personas). En su propia prueba (5.416 manos, 1.134 personas) daba 97,4 % de exactitud, 95,3 % de gestos reconocidos y 1,4 % de falsas alarmas.

**Límites de esta medición.** Son fotos, no video de una cámara web; HaGRID no tiene un pellizco puro (se usó el gesto "ok"); y las personas de HaGRID no son usuarios del visor. Falta medir con la cámara (protocolo en `eval/pruebas_usuarios/`).

## Intérprete de pedidos entrenado (respaldo del LLM)

`ml/intencion/` (vocabulario, frases y entrenamiento); corre en `backend/intent_modelo.py`. Se mide de punta a punta, con los filtros de seguridad: `python eval/eval_intent.py --frases {prueba,validacion,nuevas,objetos_validacion,objetos_prueba}`. El detalle por frase queda en `eval/resultados/intent_{reglas,modelo}[_set].csv`.

**Versión 2 (la actual): reconocer cualquier objeto.** La versión 1 se entrenó con 4.400 frases y unos 80 objetos industriales: con objetos de otro tipo se quedaba con un pedazo del nombre ("teclado mecánico" → "mecánico") o lo cortaba en "de" ("esqueleto de un dinosaurio" → "esqueleto de"). La v2 se entrena con 18.800 frases y **9.828 nombres de objetos** de WordNet en español (artefactos, máquinas, herramientas, muebles, ropa, vehículos, construcciones, animales, partes del cuerpo, frutas), con adjetivos, materiales, medidas, "con …", "para …" y "de un …" dentro del objeto, y la vista, "en 3D" o "para el cliente" afuera. Ningún objeto de entrenamiento contiene el núcleo de un objeto de los sets de medición, y el 10 % de los nombres se reservó para validar.

| Set | Frases | Quién lo vio antes | Reglas | Modelo v1 | **Modelo v2** |
|---|---|---|---|---|---|
| **Objetos, prueba (`frases_objetos_prueba.csv`)** | **45** | **Ninguno** | **60 %** | **56 %** | **91 %** |
| Objetos, validación (`frases_objetos_validacion.csv`) | 55 | La v2: se usó para elegir | 56 % | 60 % | 95 % |
| Prueba (`frases_prueba.csv`) | 38 | Las reglas: se escribieron mirándolo | 84 % | 84 % | 87 % |
| Validación (`frases_validacion.csv`) | 59 | Los dos modelos: se usó para elegirlos | 66 % | 100 % | 97 % |
| Nuevo (`frases_nuevas.csv`) | 50 | La v1 no; la v2 se armó después de verlo | 70 % | 90 % | 88 % |

En los sets de objetos un acierto pide la acción y el objeto **exacto**, con todo lo que lo describe ("silla de oficina con ruedas", "cráneo de un tigre"). Si se pide solo que el objeto traiga el núcleo, la v2 da 100 % en la prueba de objetos (reglas 67 %, v1 82 %). Los dos sets de objetos se escribieron a mano antes de entrenar, con objetos de todo tipo; el de prueba se miró por primera vez con el modelo ya elegido (dio 87 %) y una vez más después de cambiar, con la validación, el balanceo de clases del clasificador (91 %). En los sets de siempre, la v2 queda igual que la v1 (una frase más en prueba, una menos en el nuevo).

Lo que sigue fallando es casi siempre la **acción**, no el objeto: "traeme el engranaje" o "la estatua de la libertad" como analizar una pieza (el artículo "el/la" + "de la …" se parece a "mostrame el filtro de la cafetera"), "una banana" como pedido desconocido. Del objeto, se queda corto con complementos raros ("colectivo de larga distancia" → "colectivo de larga", "escritorio en l" → "escritorio").

**Etiquetador del objeto** (objeto exacto, con la acción correcta dada; validación con 1.400 frases generadas con objetos reservados y las 55 a mano):

| Etiquetador | Generadas | A mano |
|---|---|---|
| Rasgos de la palabra y sus vecinas (regresión logística, el de la v1) | 99,3 % | 96,4 % |
| Vector de la palabra en contexto (e5) + logística | 85,7 % | 89,1 % |
| Vector en contexto + red de 256 | 97,1 % | 98,2 % |
| Vector en contexto + red de 512 | 97,7 % | 98,2 % |
| **Promedio de rasgos y vector en contexto con red de 256 (elegido)** | **98,9 %** | **98,2 %** |
| Promedio de rasgos y vector en contexto con red de 512 | 99,0 % | 98,2 % |

El de rasgos es muy preciso con las formas de pedir que vio y el de contexto generaliza mejor a frases escritas por personas: el promedio toma lo mejor de los dos. La red de 512 no mejora a la de 256 (a igual resultado, la más chica).

**Clasificador de la acción** (exactitud en validación, 114 frases a mano): con e5-small, la regresión logística sin balancear las clases da 95,6 %; balanceada, con una capa de 256 o con 512-256, 94,7 %. Con MiniLM, entre 89,5 % y 92,1 %. Sin balancear gana porque "mostrar" tiene más frases y es el pedido más común: balanceada, cada frase de "analizar" pesaba casi tres veces más y empujaba "pasame el martillo" hacia "aislar". Detalle en `ml/resultados/intencion_reporte.json`.

**Tamaño y latencia:** `backend/modelos/intencion.json` pesa 1,1 MB (v1: 120 KB). Se descartaron los rasgos con peso menor a 0,05 (70 mil → 13 mil): no cambia ningún objeto de validación y ahorra 1,6 MB. 13 ms de mediana por frase en CPU (contra 2 segundos o más del LLM): una sola pasada por el modelo de embeddings da el vector de la frase y el de cada palabra.

## Latencias

### Backend (`eval/medir_latencias.py`)

50 pedidos: las 10 frases de "mostrar" del set de intención, 5 veces cada una. Backend local con búsqueda híbrida, sin LLM y sin generación.

| Etapa | Mediana | p95 |
|---|---|---|
| Interpretar el pedido (`/intent`, intérprete por reglas) | 1,7 ms | 1,9 ms |
| Buscar el objeto (`/object`) | 1,1 ms | 1,3 ms |
| Descargar el `.glb` | 11,7 ms | 12,9 ms |
| **Total sin dibujar** | **14,2 ms** | **15,4 ms** |

35 de los 50 pedidos terminaron en un modelo. Los otros 15 son las 3 frases que el intérprete por reglas no entiende y "bomba de agua", que no está en el catálogo (con la generación desactivada).

### De punta a punta en el navegador

El visor registra cuánto tarda cada pedido, desde que se envía hasta que el modelo se **dibuja** en pantalla, y cuánto tarda cada fotograma de la cámara. Para verlo:

1. Abrir el visor, apretar `D` para abrir el panel de depuración y hacer 10 pedidos del catálogo.
2. Leer las líneas `pedido_catalogo` y, con cámara, `fotograma` (mediana, p95 y cantidad).
3. Para exportar los valores, escribir `copy(metricas.csv())` en la consola del navegador.

**Pendiente:** no se pudo medir en el entorno de desarrollo. El navegador que se usó tenía la ventana oculta y, en ese estado, Chrome pausa el dibujo. Se verificó que el registro funciona; los números los tiene que tomar alguien con la ventana visible.

## Búsqueda

### Validación con un segundo set

El umbral (0,62) se eligió con `eval/busqueda_prueba.csv`. Para medirlo sin ese sesgo, se escribió `eval/busqueda_validacion.csv`: 20 consultas nuevas, 13 que están en el catálogo y 7 que no.

| | Calibración | Validación |
|---|---|---|
| Híbrido, versión inicial | 90 % | **75 %**: por debajo de la meta |
| Híbrido con la regla de palabras sueltas | 90 % | **90 %** |

**Qué pasó.** Probando el visor, "mostrame una silla" mostraba el tornillo. Las consultas de una sola palabra que no están en el catálogo ("silla", "mesa", "zapatilla") superaban el umbral con los embeddings. El set de calibración no lo detectaba porque todos sus negativos tenían varias palabras.

**Corrección: una regla, no un umbral ajustado.** Si la consulta tiene una sola palabra de contenido y no coincide con ningún nombre ni sinónimo, no se usan embeddings y el objeto se trata como nuevo. Los embeddings de palabras aisladas no tienen contexto, y las paráfrasis que resuelven siempre tienen varias palabras.

**Advertencia honesta.** La falla se descubrió antes de escribir el set de validación, y por eso ese set incluye negativos de una palabra. El 90 % de validación no es tan independiente como el 75 % inicial.

**Errores que quedan** (no se corrigieron, para no ajustar el sistema al set de validación):
- "pieza roscada para unir dos chapas" → engranaje (debería ser el tornillo);
- "taladro percutor" → engranaje (no está en el catálogo, similitud 0,80).

### Comparación de métodos (set de calibración)

| Método | Exactitud | Dónde falla |
|---|---|---|
| Palabras clave | 80 % | Paráfrasis sin palabras en común ("máquina que convierte electricidad en movimiento") |
| Embeddings MiniLM | 75 % (umbral 0,45) | Sinónimos técnicos raros ("cojinete de bolas" → tornillo) |
| Embeddings E5-small | 75 % (umbral 0,85) | Da similitud alta a todo y no rechaza lo que no está en el catálogo |
| **Híbrido (palabras + MiniLM)** | **90 %** (umbral 0,62) | "pieza para regular el caudal de una cañería" → engranaje; "elemento de fijación roscado" → no encontrado |
