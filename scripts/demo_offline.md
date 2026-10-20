# Modo sin internet (plan B)

Objetivo: poder mostrar gestos + visor + catálogo aunque falle la red del aula.

## Antes de la defensa
- [ ] Descargar Three.js y MediaPipe (wasm + `hand_landmarker.task`) a `frontend/vendor/` y apuntar el importmap ahí.
- [ ] Cargar los `.glb` del catálogo en `frontend/models/`.
- [ ] Descargar el modelo de `sentence-transformers` (queda en caché tras el primer uso).
- [ ] Tener el video de respaldo del demo en la notebook y en un pendrive.

## Durante la defensa
1. Levantar backend y frontend (ver README).
2. Correr `scripts/check_all.sh`.
3. Pedir solo objetos que estén en el catálogo; el LLM remoto no se usa.
