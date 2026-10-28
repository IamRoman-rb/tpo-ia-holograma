# Resultados de la evaluación

Ver el plan completo en la sección 15 de [propuesta.md](propuesta.md). Para reproducir los números (desde la raíz):

```bash
python eval/eval_intent.py                                  # intención: reglas y LLM (el LLM necesita ANTHROPIC_API_KEY)
python eval/eval_busqueda.py                                # búsqueda: comparación de métodos y barrido de umbral
python eval/eval_busqueda.py --set validacion --solo-producto   # búsqueda en el set que no se usó para calibrar
python eval/medir_latencias.py                              # latencias del backend (con LIMITE_PEDIDOS_MINUTO=1000)
python eval/pruebas_usuarios/resumir.py                     # totales de las pruebas con usuarios
```

Los detalles por frase y por pedido quedan en `eval/resultados/`.

| Área | Métrica | Meta | Resultado | Notas |
|---|---|---|---|---|
| Gestos | Precisión por gesto (persona no vista) | ≥ 90 % | **MLP: 95,3 %** (recall medio de los 7 gestos, fotos de HaGRID) · con cámara: pendiente | Ver "Gestos con datos de internet" abajo |
| Gestos | Reglas vs. MLP | Reportar | **Exactitud 63,2 % vs. 97,4 %** · falsas alarmas 53,8 % vs. 1,4 % | 5.416 manos de personas que el modelo no vio |
| Gestos | Falsos positivos en 60 s en reposo | ≤ 2 | Pendiente | Incluido en el protocolo de pruebas; necesita cámara |
| Latencia | Gesto → pantalla | < 100 ms | Pendiente | Con cámara: panel de depuración (tecla D), línea `fotograma` |
| Latencia | Pedido → modelo (catálogo) | < 2 s | **Backend: 14 ms** (mediana; p95 15 ms) · de punta a punta: pendiente | Medido con el **intérprete por reglas**. Con el LLM, la interpretación va a tardar del orden de segundos: hay que medirlo |
| Latencia | Pedido → modelo (generado) | Reportar | Pendiente | El notebook de Colab mide TripoSR y Shap-E |
| Generación | Modelos por piezas válidos, partes clave y calidad (10 pedidos) | Reportar | Pendiente | `eval/eval_ensamble.py`; necesita clave de API. Mide también segundos y costo por modelo |
| LLM | Aciertos de intención (38 frases) | ≥ 85 % | **Reglas: 82 % (31/38)** · LLM: pendiente | El LLM se evalúa cuando haya una clave de API. Se sumaron 4 frases de desarmado y 4 de análisis de piezas; con las 30 originales las reglas daban 80 % (24/30) |
| Búsqueda | Recuperación correcta (20 pedidos) | ≥ 80 % | **Híbrido: 90 % en calibración y 90 % en validación** | Ver el detalle abajo |
| Usabilidad | Éxito de tarea sin ayuda | ≥ 80 % | Pendiente | Protocolo con 5 tareas en `eval/pruebas_usuarios/protocolo.md` |

## Gestos con datos de internet (HaGRID)

`ml/extraer_hagrid.py` y `ml/entrenar_mlp.py`. Fuente: muestra de 31.833 fotos de HaGRID (Hugging Face `cj-mills/hagrid-sample-30k-384p`, CC BY-SA 4.0). MediaPipe encontró y asignó **34.503 manos de 7.798 personas**. Partición fija por persona: 24.328 manos para entrenar, 4.759 para validar y **5.416 para la prueba (1.134 personas que el modelo no vio)**.

| Método | Exactitud | F1 macro | Gestos reconocidos (recall medio) | Falsas alarmas (manos sin gesto tomadas como gesto) |
|---|---|---|---|---|
| Reglas (`gestures.js`) | 63,2 % | 0,699 | 93,1 % | **53,8 %** |
| **MLP** (128-64, umbral 0,9) | **97,4 %** | **0,963** | **95,3 %** | **1,4 %** |

**Por clase (prueba, MLP):** palma 94 %, pellizco 99 %, pulgar abajo 96 %, pulgar arriba 97 %, puño 97 %, señalar 93 %, victoria 91 %, ninguno 99 %. Matrices de confusión: `ml/resultados/hagrid_{mlp,reglas}_confusion.csv`.

**Qué muestra la comparación.** Las reglas encuentran casi todos los gestos, pero también "ven" gestos donde no los hay: una mano con cuatro dedos extendidos es una palma para ellas, y dos dedos juntos hacia arriba, una victoria. En el visor, cada falsa alarma es una acción que nadie pidió. El MLP aprendió esas diferencias de miles de ejemplos.

**Sesgo por mano:** el MLP reconoce los gestos igual con la mano derecha (95,3 %) que con la izquierda (95,1 %). **Por persona:** de 177 personas con al menos 3 gestos en la prueba, la mediana es 100 % y 8 quedan por debajo del 80 %.

**Límites de esta medición.** Son fotos, no video de una cámara web; HaGRID no tiene un pellizco puro (se usó el gesto "ok"); y las personas de HaGRID no son usuarios del visor. Falta medir con la cámara (protocolo en `eval/pruebas_usuarios/`).

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
