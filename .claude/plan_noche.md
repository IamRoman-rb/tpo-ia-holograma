# Plan de desarrollo nocturno (27→28/10/2026)

Pedido del usuario (textual, resumido): "Seguí desarrollando los modelos, entrenándolos y perfeccionándolos.
Agregar interfaz por voz, nuevos gestos con las manos y demás. Poder seleccionar una pieza por separado
con los dedos y separarla para analizarla. Entrenar los modelos con imágenes de internet para identificar
patrones. Las PPT/presentación: dejarlas totalmente de lado."

Reglas que siguen valiendo:
- No commitear ni pushear sin que el usuario lo pida (al final, dejar todo listo y explicar qué va en cada commit).
- No salirse del eje de la propuesta (C:\Users\USER\Downloads\propuesta_tpo_holograma_ia.md); señalar desvíos.
- Nunca commitear .env; .claude/ queda sin commitear.
- Responder en español rioplatense. Tests: backend/.venv (pytest), node frontend/tests/*.mjs.
- Servidores de preview: .claude/launch.json (frontend 5500, backend 8000).

Rama de trabajo: feature/modelos (sale de feature/presentacion = feature/eval-ensamble + guion sin commitear).
OJO: docs/presentacion/guion.md y la línea del README que lo enlaza son de la presentación (no tocar más).

## Tareas (marcar [x] al terminar, anotar notas)

1. [ ] Selección de piezas con los dedos + aislar para analizar
   - Señalar: cursor que sigue la punta del índice; la pieza bajo el cursor se resalta (raycast).
   - Pellizco sobre una pieza resaltada: la toma y la saca del conjunto (arrastrar solo esa pieza).
   - Gesto/acción "aislar": la pieza elegida queda sola, centrada y grande; las demás se atenúan; panel de análisis
     (nombre, descripción, material, medidas aprox.). Salir: puño o "volver".
   - Pedidos: "mostrame solo el volante", "aislá el impulsor", "volvé al conjunto".
   - Antes "señalar" = desmontar siguiente: reasignar (documentar en decisiones.md).
2. [ ] Gestos nuevos (reglas): pulgar arriba (confirmar), victoria/dos dedos, etc. + tests sintéticos + captura.html.
3. [ ] Interfaz por voz: Web Speech API (es-AR), botón micrófono + tecla; aviso de privacidad (audio a un servicio
   externo salvo reconocimiento local); mismo flujo que el texto. Respuesta hablada opcional.
4. [ ] Dataset de internet para gestos: buscar dataset público con licencia (HaGRID u otro), descargar una muestra,
   extraer landmarks con MediaPipe (mismo .task del frontend), normalizar igual que el navegador, entrenar MLP,
   evaluar dejando personas afuera, comparar con las reglas (gestures.js vía node). Registrar fuente, licencia, tamaño.
5. [ ] MLP en el navegador: pesos exportados (JSON) e inferencia en JS; selector Reglas/MLP en el visor.
6. [ ] Documentar: decisiones.md, metricas.md (reglas vs MLP con datos de internet), uso_responsable.md (voz, dataset),
   README (gestos y voz).

## Notas de avance

- 27/10 23:40 · Rama feature/modelos creada. Dataset elegido: HaGRID sample 30k 384p (Hugging Face cj-mills/hagrid-sample-30k-384p,
  CC BY-SA 4.0 variante HaGRID, 31.833 imágenes, 18 clases + no_gesture, anotaciones con bboxes, labels, leading_hand, user_id).
  DESCARGADO: C:\Users\USER\Documents\proyectos\datasets\hagrid-sample-30k-384p\hagrid-sample-30k-384p.zip (1,0 GB). Falta descomprimir
  (en esa misma carpeta, fuera del repo).
- Entorno ml/.venv-mediapipe (Python 3.12 de uv) con mediapipe + pandas + sklearn + skl2onnx + onnxruntime: la instalación quedó
  INSTALADO OK (import mediapipe imprime versión 1.2.0, sklearn 1.9.1; revisar que exista mediapipe.tasks.python.vision.HandLandmarker).
  Agregar `.venv-*/` al .gitignore.
- Diseño decidido para gestos e interacción:
  · Mapeo HaGRID → clases: palm/stop → palma_abierta; fist → puño; one → señalar; ok → pellizco (proxy, documentarlo);
    like → pulgar_arriba (nuevo); dislike → pulgar_abajo (nuevo); peace → victoria (nuevo); resto (call, four, three, rock, mute,
    two_up, *_inverted, three2) y no_gesture → ninguno.
  · Extracción: MediaPipe HandLandmarker (frontend/vendor/.../hand_landmarker.task) sobre cada imagen; asignar mano detectada al bbox
    anotado; guardar CSV con el formato de ml/preparar.py (clase, persona=user_id, mano, aspecto, x0..z20) en ml/data/externo/ (gitignored).
  · Evaluar con GroupKFold por user_id; comparar con las reglas (ml/evaluar_reglas.mjs) en las mismas muestras.
  · Interacción: señalar = cursor en la punta del índice + resaltado de la pieza (raycast) y selección por permanencia (~0,8 s);
    pellizco sobre pieza seleccionada = tomar y sacar SOLO esa pieza; victoria con pieza seleccionada = aislar/analizar
    (sin selección = alternar despiece); puño = salir del análisis o reiniciar; pulgar arriba = confirmar; pulgar abajo = deshacer.
    Desmontar siguiente queda en botón, tecla X y pedido escrito.
- Próximo paso: descomprimir el zip, escribir ml/extraer_hagrid.py, y en paralelo arrancar la tarea 1 (selección de piezas) en el frontend.
- 28/10 02:35 · HECHO: zip descomprimido; ml/extraer_hagrid.py → ml/data/externo/hagrid.csv (34.503 manos, 7.798 personas,
  2.281 imágenes sin mano asignada). preparar.py: CLASES + pulgar_arriba/pulgar_abajo/victoria y particion(persona) por hash
  (15 % prueba / 15 % validación / 70 % entrenamiento); CSV por partición en ml/data/externo/{prueba,entrenamiento}/.
  gestures.js: medir() exportada con pulgarAfuera, pulgarVertical, separacionV; reglas de pulgar arriba/abajo y victoria
  calibradas con ml/explorar_metricas.mjs (solo entrenamiento). .gitignore: .venv-*/ y ml/data/externo/.
  Reglas en PRUEBA (personas no vistas): exactitud 63,2 %; recall gestos 85-98 %, pero "ninguno" recall 46 % (913 → palma,
  487 → victoria, 226 → señalar). Precisión: palma 35 %, victoria 32 %, señalar 49 %.
- Visor (tarea 1, en curso): viewer.js reescrito con apuntar(u,v,progreso) + cursor DOM .cursor-mano (FALTA CSS), moverPieza,
  aislar/salirDeAislamiento/infoPieza/buscarPieza/seleccionada; undo incluye libres y aislada. piezas.js: encuadreAislada,
  nombreMaterial, textoMedidas, crearSeleccionPorPermanencia (FALTAN tests). geometria.a_glb(unidad="cm") para ensamble.
  piezas.test.mjs: +14 checks OK.
- 28/10 02:55 · MLP ENTRENADO (ml/entrenar_mlp.py): elegido (128,64) alpha 1e-3, umbral confianza 0,9 (con validación).
  PRUEBA (5.416 manos, personas no vistas): MLP exactitud 97,4 %, F1 macro 0,963, recall gestos 95,3 %, falsas alarmas 1,4 %;
  REGLAS 63,2 % / 0,699 / 93,1 % / 53,8 %. Reporte: ml/resultados/hagrid_mlp_reporte.json + confusiones CSV.
  Exportado frontend/ml/mlp_gestos.json (332 KB); frontend/js/mlp.js (inferencia JS) + tests/mlp.test.mjs OK (dif 5.8e-6).
  gestures.js: crearClasificador({ mlp }). FALTA: selector Reglas/MLP en el visor (main.js) y cargar el modelo.
- 28/10 03:10 · controles.js reescrito (tabla de gestos en la cabecera; alEvento: seleccion/aislar/salir/despiece/confirmar/
  deshacer); manos_sinteticas.js con poses pulgar_arriba/pulgar_abajo/victoria; gestos.test.mjs 47 OK (tests 9 y 10 nuevos).
  YA NO FALTA controles.js. SIGUE FALTANDO lo de abajo salvo controles.
- 28/10 03:40 · TAREA 1 CASI COMPLETA: index.html (selector Reglas/MLP en el HUD, botón Analizar, panel #analisis, ayuda de
  gestos), ui.js (mostrarAnalisis, confirmarPendiente, nombres), main.js (selector con localStorage, MLP por defecto, simulación
  siempre con reglas porque el MLP no reconoce manos sintéticas; EVENTOS_GESTO; acciones aislar/conjunto; teclas I y Esc;
  ejecutar() con acciones 'aislar'/'conjunto'), CSS (cursor, panel, selector; #piezas top 252px). Verificado en el navegador:
  aislar/volver con teclado y flujo con dedos cuadro a cuadro (señalar 0,8 s → elige volante; pellizco → 1/7 afuera; victoria →
  aislada). Simulación ampliada con ese flujo. Con el MLP: pellizco iniciado se mantiene por histéresis de distancia.
  FALTA: backend intents 'aislar'/'conjunto' (schemas, prompt, reglas, tests, frases) → luego voz → docs.
- 28/10 04:05 · HECHO intents aislar/conjunto (schemas, prompt, intent.py _pieza/RE_AISLAR/RE_CONJUNTO, 5 tests, 4 frases nuevas;
  eval reglas 31/38 = 82 %; falla honesta "quiero ver el impulsor solo"). main.js: aislar con pieza inexistente → mostrar objeto.
  Verificado en navegador: "mostrame solo el volante" aísla; "volvé al conjunto" vuelve.
  HECHO VOZ (tarea 3): frontend/js/voz.js (Web Speech es-AR, parciales, errores claros, processLocally si available(),
  decir() con speechSynthesis solo si el pedido vino por voz) + tests/voz.test.mjs (12 OK). index.html botón #microfono + tecla V
  + aviso de privacidad de voz en la ayuda; CSS latido. Verificado: botón visible en Chromium; sin permiso → mensaje claro.
  No se pudo probar con micrófono real.
  FALTA: docs (decisiones, metricas, uso_responsable, README, ml/README) + correr todos los tests + resumen final.
- 28/10 04:30 · TODO TERMINADO. Docs: decisiones (12 entradas), metricas (sección HaGRID, filas de gestos y LLM 31/38),
  uso_responsable (voz, dataset, sesgo por mano: 95,3 % der. vs 95,1 % izq.; 8/177 personas < 80 %), README (gestos, análisis,
  voz, MLP), ml/README (cómo reproducir), arquitectura (archivos). check_all.sh corre piezas/mlp/voz.
  Tests: frontend gestos/piezas/mlp/voz OK · backend 64 · ml 4. Nada commiteado.
  Tareas 1-6: [x] todas. Pendiente para el usuario: probar con cámara y micrófono reales; commits.
  FALTA: controles.js (señalar→apuntar+permanencia, pellizco sobre pieza elegida, victoria, pulgares), main.js/ui.js (panel de
  análisis, pedidos "aislá X"/"volvé"), CSS del cursor, intents nuevos en backend (aislar).
