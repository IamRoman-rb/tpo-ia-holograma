# Evaluación de generadores 3D

Sección 5.4 de la propuesta: elegir el generador con criterios medibles (calidad, tiempo y costo) antes de integrarlo al backend.

## Cómo correrlo

1. Abrir `evaluar_generadores_colab.ipynb` en Google Colab, de una de estas dos formas:
   - **Archivo → Subir notebook**;
   - con el enlace `https://colab.research.google.com/github/IamRoman-rb/tpo-ia-holograma/blob/<rama>/eval/generacion3d/evaluar_generadores_colab.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)**. La versión gratuita alcanza.
3. **Entorno de ejecución → Ejecutar todas**. Tarda entre 20 y 30 minutos, sobre todo en instalar y descargar los modelos.
4. Se descarga `resultados_generacion3d.zip`. Descomprimirlo en `eval/generacion3d/resultados/`.

## Qué compara

| Generador | Flujo | Objetos |
|---|---|---|
| TripoSR | texto → imagen (SDXL-Turbo) → 3D | 3 piezas del catálogo y 2 que no están (llave inglesa, bomba de agua) |
| Shap-E | texto → 3D | las mismas 5 |

Métricas automáticas: segundos por modelo, tiempo de carga en frío, vértices, caras, tamaño del `.glb` y si la malla es estanca.

## Puntuar la calidad (la hacen personas)

2 o 3 integrantes completan en `resultados.csv`, de 1 a 5:

| Puntaje | `calidad_forma`: ¿se reconoce el objeto? | `calidad_detalle`: ¿tiene las partes importantes? |
|---|---|---|
| 1 | No se reconoce | Sin partes reconocibles o con artefactos graves |
| 3 | Se reconoce con esfuerzo | Faltan partes o hay deformaciones visibles |
| 5 | Se reconoce al instante | Partes principales correctas, sin artefactos |

Para juzgar, conviene abrir los `.glb` en el visor del proyecto (copiarlos a `frontend/models/`) o en <https://gltf-viewer.donmccurdy.com/>. Las vistas previas del notebook son aproximadas.

Con los puntajes se completa la tabla de decisión del final del notebook y se registra la elección en `docs/decisiones.md`.

## Generación por piezas con el LLM

El método que usa el visor por defecto (`GEN3D_MODO=ensamble`) no necesita Colab y se evalúa desde la raíz, con el entorno del backend y `ANTHROPIC_API_KEY` en `.env`:

```bash
python eval/eval_ensamble.py                  # 10 pedidos = 10 llamadas a la API (tiene costo)
python eval/eval_ensamble.py --solo morsa     # uno solo, para probar
python eval/eval_ensamble.py --prueba         # sin API ni costo: solo verifica que el circuito funcione
```

Los pedidos están en `eval/pedidos_ensamble.csv`. Los cinco primeros (`en_colab = si`) son los mismos objetos que se generan con TripoSR y Shap-E, así que los tres métodos se comparan con la misma rúbrica.

Métricas automáticas, en `eval/resultados/ensamble.csv`:

| Columna | Qué mide |
|---|---|
| `ok`, `error` | Si se generó un modelo válido; si no, por qué (rechazo, respuesta cortada, diseño inválido) |
| `segundos` | Tiempo de la llamada al LLM (el armado del `.glb` tarda milisegundos) |
| `modelo`, `tokens_entrada`, `tokens_salida`, `costo_usd` | Qué modelo respondió (puede ser uno de respaldo) y cuánto costó, con los precios de octubre de 2026 |
| `piezas`, `desmontables`, `fijas` | Cuántas piezas tiene y cuántas se pueden sacar |
| `formas_descartadas` | Formas que la validación rechazó (medidas fuera de rango, tubos con el agujero más grande que el tubo…) |
| `partes_clave` | Cuántas de las partes que no pueden faltar aparecen en los nombres de las piezas (por ejemplo, en la bomba: carcasa, impulsor, motor, eje y base). Es una aproximación: acepta sinónimos, pero un nombre raro cuenta como faltante |
| `medida_mayor_cm`, `escala_ok` | Lado mayor del modelo y si está en un rango razonable para ese objeto (detecta errores de unidades) |
| `kb` | Peso del `.glb` |

Los modelos quedan en `frontend/models/evaluacion/`. Con el frontend levantado, cada uno se abre en el visor con la URL de la columna `ver`, y ahí se puede desarmar.

### Puntuar la calidad

Igual que con TripoSR y Shap-E, 2 o 3 integrantes completan `calidad_forma` y `calidad_detalle` con la rúbrica de arriba. Para este método se agrega una columna:

| Puntaje | `desarmado_logico`: ¿el orden y las direcciones tienen sentido? |
|---|---|
| 1 | Las piezas salen en un orden imposible o se atraviesan |
| 3 | El orden es razonable, pero alguna pieza sale hacia un lado que no corresponde o queda trabada |
| 5 | Se desarma como un mecánico lo haría: primero lo exterior, cada pieza por donde puede salir |

Si se vuelve a correr un pedido, su modelo cambia y sus puntajes se borran. Los de los pedidos que no se corrieron se conservan.
