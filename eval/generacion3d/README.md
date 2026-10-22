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
