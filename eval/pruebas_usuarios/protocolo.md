# Protocolo de pruebas con usuarios externos (sección 15.2)

**Objetivo:** medir si una persona que no conoce el proyecto puede usar el visor sin ayuda (meta: ≥ 80 % de tareas completadas), y medir la precisión de los gestos con personas que **no** grabaron el dataset (meta: ≥ 90 %).

## Participantes

- 3 a 5 personas **ajenas al grupo**, con perfiles distintos: edad, mano dominante y familiaridad con la tecnología.
- Nadie del grupo y nadie que haya grabado datos para el MLP.
- Identificarlas con un código (`U1`, `U2`, …), nunca por nombre.

## Antes de empezar (5 minutos)

1. Correr `bash scripts/check_all.sh`: no tiene que haber ninguna línea `FALLA`.
2. Anotar las condiciones: luz (buena o baja), fondo (liso o con detalle), cámara (integrada o externa).
3. Leer el consentimiento:
   > "Vamos a probar un visor 3D que se maneja con las manos. Probamos el sistema, no a vos: si algo no sale, es un problema del sistema. La cámara se procesa en esta computadora y no se graba nada. Anotamos solo si completás las tareas y cuánto tardás, con un código en lugar de tu nombre. ¿Estás de acuerdo?"

## Tareas (sin explicar cómo se hace)

Leer la consigna textual. **No ayudar ni dar pistas**: si a los 2 minutos no lo logró, anotar "no completó" y pasar a la siguiente.

| # | Consigna textual |
|---|---|
| T1 | "Pedí un objeto y mostrame su parte de atrás." (consigna de la sección 15.2) |
| T2 | "Agrandalo con las manos." |
| T3 | "Movelo hacia un costado." |
| T4 | "Volvé a dejarlo como estaba al principio." |
| T5 | "Pedí un objeto que no esté en la lista de la derecha." |

Para cada tarea, anotar en `planilla_tareas.csv`:
- si la completó (`si`, `no` o `con_ayuda`);
- los segundos desde que termina la consigna hasta que lo logra;
- los errores (acciones que no quería) y dónde dudó;
- qué gesto probó que el sistema no reconoce.

## Precisión de gestos (10 minutos)

Abrir el panel de depuración con la tecla `D`. Pedirle a la persona que haga cada gesto **20 veces**: mano en reposo → gesto → reposo. Anotar en `planilla_gestos.csv` cuántas veces el panel mostró el gesto correcto.

| Gesto | Cómo pedirlo |
|---|---|
| pellizco | "Juntá el pulgar y el índice." |
| palma_abierta | "Mostrame la palma abierta." |
| puño | "Cerrá la mano." |
| señalar | "Señalá con el índice." |
| escala_dos_manos | "Con las dos manos, juntá pulgar e índice." |

**Falsos positivos (sección 15.1, meta ≤ 2):** con la mano en reposo frente a la cámara durante 60 segundos, contar cuántas veces el objeto se movió solo.

## Al final: tres preguntas

1. ¿Qué te resultó más natural?
2. ¿Qué te resultó confuso?
3. ¿Lo usarías para mostrarle un producto a un cliente? ¿Por qué?

## Después

- Guardar las planillas completas en `eval/resultados/pruebas_usuarios/`.
- Pasar los totales a `docs/metricas.md` (usabilidad y gestos) y a la tabla de sesgos por condición de `docs/uso_responsable.md`.
- Usar las respuestas para las mejoras de último momento y para la conclusión de la defensa.
