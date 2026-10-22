# Resultados de la evaluación

Ver el plan completo en la sección 15 de [propuesta.md](propuesta.md). Para reproducir los números:

```bash
python eval/eval_intent.py      # intención: reglas y LLM (el LLM necesita ANTHROPIC_API_KEY)
python eval/eval_busqueda.py    # búsqueda en el catálogo + barrido de umbral
```

Los detalles por frase quedan en `eval/resultados/`.

| Área | Métrica | Meta | Resultado | Notas |
|---|---|---|---|---|
| Gestos | Precisión por gesto (persona no vista) | ≥ 90 % | Pendiente | Necesita cámara y dataset |
| Gestos | Reglas vs. MLP | Reportar | Pendiente | El notebook `ml/train_mlp.ipynb` ya genera la comparación; falta grabar el dataset real |
| Gestos | Falsos positivos en 60 s en reposo | ≤ 2 | Pendiente | Necesita cámara |
| Latencia | Gesto → pantalla | < 100 ms | Pendiente | El panel de depuración (tecla D) muestra el tiempo de procesamiento por fotograma |
| Latencia | Pedido → modelo (catálogo) | < 2 s | Pendiente de medición formal | Observado por debajo de 2 s con el backend precargado |
| Latencia | Pedido → modelo (generado) | Reportar | Pendiente | Generación 3D no integrada |
| LLM | Aciertos de intención (30 frases) | ≥ 85 % | **Reglas: 80 % (24/30)** · LLM: pendiente | El LLM se evalúa cuando haya una clave de API |
| Búsqueda | Recuperación correcta (20 pedidos) | ≥ 80 % | **Híbrido: 90 % (18/20)** | Palabras 80 % · MiniLM 75 % · E5 75 % (cada uno en su mejor umbral) |
| Usabilidad | Éxito de tarea sin ayuda | ≥ 80 % | Pendiente | Fase 4 |

## Búsqueda: detalle de la comparación

| Método | Exactitud | Dónde falla |
|---|---|---|
| Palabras clave | 80 % | Paráfrasis sin palabras en común ("máquina que convierte electricidad en movimiento") |
| Embeddings MiniLM | 75 % (umbral 0.45) | Sinónimos técnicos raros ("cojinete de bolas" → tornillo) |
| Embeddings E5-small | 75 % (umbral 0.85) | Da similitud alta a todo y no rechaza lo que no está en el catálogo |
| **Híbrido (palabras + MiniLM)** | **90 %** (umbral 0.62) | "pieza para regular el caudal de una cañería" → engranaje; "elemento de fijación roscado" → no encontrado |

Advertencia: el umbral se eligió con las mismas 20 consultas con las que se mide. Antes de la defensa conviene medirlo con un segundo set que no se haya usado para calibrar.
