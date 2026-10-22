# Modo sin internet (plan B, riesgo 2)

Objetivo: poder mostrar el visor, los gestos y el catálogo aunque falle la red del aula. "Si no abre, no se presentó."

## Qué funciona sin internet

| Parte | Sin internet | Por qué |
|---|---|---|
| Visor 3D y detección de manos | ✅ | Three.js, MediaPipe y el modelo de manos están en `frontend/vendor/` |
| Catálogo y búsqueda | ✅ | El modelo de embeddings queda en la caché de Hugging Face después del primer uso |
| Modelos generados antes | ✅ | Quedan guardados en `frontend/models/generados/` y en el catálogo |
| Interpretación del pedido | 🔶 | Sin red, el LLM no responde y se usa el intérprete por reglas (80 % en el set de prueba) |
| Generar modelos nuevos | ❌ | El servidor de generación corre en Colab |

## Antes de la defensa

- [ ] Clonar el repositorio en una **segunda computadora** y levantar todo con el README.
- [ ] Levantar el backend una vez **con internet**, para que se descargue el modelo de embeddings.
- [ ] Generar de antemano los objetos que se van a pedir en la demo: quedan en el catálogo y salen al instante aunque el servidor de Colab no esté (riesgo 1).
- [ ] Ensayar la demo con frases que el intérprete por reglas entiende bien (empezar con "mostrame…").
- [ ] Tener el video de respaldo del demo en la notebook y en un pendrive.

## Durante la defensa

1. Levantar backend y frontend (ver README). Si no hay red, arrancar el backend con `HF_HUB_OFFLINE=1` para que no intente conectarse a Hugging Face:
   ```bash
   cd backend && HF_HUB_OFFLINE=1 uvicorn main:app --port 8000
   ```
2. Correr la verificación:
   ```bash
   bash scripts/check_all.sh
   ```
   Si no aparece ninguna línea `FALLA`, la demo funciona. Los `AVISO` indican funciones que faltan, como el LLM o la generación.
3. Sin red, pedir solo objetos que ya estén en el catálogo.
