# Modo sin internet (plan B)

Objetivo: poder mostrar gestos + visor + catálogo aunque falle la red del aula.

## Antes de la defensa
- [ ] Descargar Three.js y MediaPipe (wasm + `hand_landmarker.task`) a `frontend/vendor/` y apuntar el importmap ahí.
- [ ] Cargar los `.glb` del catálogo en `frontend/models/`.
- [ ] Levantar el backend una vez con internet: descarga el modelo de embeddings y queda en caché.
- [ ] Para la defensa, arrancar el backend con `HF_HUB_OFFLINE=1`: si no, al iniciar consulta a Hugging Face y puede demorarse sin red.
- [ ] Sin clave de API o sin internet, la intención la resuelve el intérprete por reglas (80 % en el set de prueba). Ensayar la demo con frases que las reglas entienden bien.
- [ ] Tener el video de respaldo del demo en la notebook y en un pendrive.

## Durante la defensa
1. Levantar backend y frontend (ver README).
2. Correr `scripts/check_all.sh`.
3. Pedir solo objetos que estén en el catálogo; el LLM remoto no se usa.
