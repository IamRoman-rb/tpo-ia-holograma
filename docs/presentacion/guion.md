# Guion de la defensa (15 minutos)

Borrador para ensayar. Sigue la sección 20 de la propuesta: **primero la conclusión, después cómo se llegó**. Todos exponen y cada uno presenta su rol (sección 11). Las diapositivas están en <https://claude.ai/artifact/SGqE5T7pAnKSBRRLnZ8UDA> (13 diapositivas, con notas para quien habla; se descargan como PowerPoint o PDF). Los números marcados como **⟨pendiente⟩** se completan cuando estén medidos (ver [metricas.md](../metricas.md)); si no se llegan a medir, se dice que son pendientes, no se inventan.

**Frase de propuesta** (se dice igual al principio y al final):

> Un visor 3D que se controla con las manos y se crea con una frase, para que los equipos comerciales y técnicos muestren productos o piezas a un cliente sin mouse ni modelado 3D.

## Tiempos

| Min | Bloque | Quién | Diapositiva |
|---|---|---|---|
| 0:00–1:30 | Conclusión adelantada y frase de propuesta | Rol 1 | 1–2 |
| 1:30–3:00 | Producto, usuario y problema (el antes) | Rol 1 | 3–4 |
| 3:00–5:00 | Técnicas de IA: cuáles y por qué | Roles 2 y 3 | 5–7 |
| 5:00–6:30 | Arquitectura: qué entra, qué decide, qué ve el usuario | Rol 5 | 8 |
| 6:30–11:00 | **Demo en vivo** | Rol 4 conduce; Roles 2 y 3 comentan | 9 |
| 11:00–13:00 | Métricas, limitaciones y uso responsable | Rol 5 | 10–11 |
| 13:00–15:00 | Valor para el negocio y respuesta a la propuesta | Rol 1 | 12–13 |

Si el grupo es de 6, el rol 5b toma el bloque de 11:00 a 13:00.

## Qué decir en cada bloque

### 0:00–1:30 · Conclusión adelantada (Rol 1)

- "Les adelanto la conclusión: **un vendedor puede mostrar en una reunión una pieza que no tiene modelada, pidiéndola con una frase, y desarmarla con la mano**."
- Una prueba en números: un modelo del catálogo aparece en ⟨pendiente: de punta a punta⟩ (el backend responde en 14 ms); uno nuevo se genera en ⟨pendiente: eval_ensamble⟩ segundos y ya viene por piezas.
- Leer la frase de propuesta.

### 1:30–3:00 · El antes (Rol 1)

- Historia: "Carla tiene una reunión en 10 minutos y necesita mostrar un producto que no tiene modelo 3D. Antes llamaba a un diseñador y esperaba días."
- El antes (sección 3.4): modelo nuevo en horas o días, se muestra con mouse o fotos, hace falta saber software 3D.
- **Aclarar que esos tiempos son estimaciones** si no se consiguieron entrevistas.
- Usuario: Gerencia Comercial y Gerencia Técnica. No son especialistas en IA.

### 3:00–5:00 · Técnicas de IA (Roles 2 y 3)

Mensaje: cada técnica resuelve un problema distinto del producto, no se usan muchas por usar.

- **Rol 2 · Visión y gestos:** MediaPipe da 21 puntos por mano, en el navegador; el video nunca sale del equipo. Los gestos se reconocen con reglas explicables (relaciones entre dedos, normalizadas por el tamaño de la mano). El MLP se entrena con el dataset del grupo y se compara contra las reglas: ⟨pendiente: reglas vs. MLP⟩.
- **Rol 3 · Lenguaje:** Claude convierte la frase en un JSON con esquema fijo. Si no hay red o falla, un intérprete por reglas sigue funcionando (82 % de aciertos sin LLM; meta con LLM: 85 %, ⟨pendiente⟩).
- **Rol 3 · Búsqueda:** primero palabras clave, después embeddings multilingües. 90 % de recuperación correcta, también en un set que no se usó para calibrar.
- **Rol 3 · Generación por piezas:** cuando el objeto no está, Claude lo diseña como un ensamble de piezas (cilindros, engranajes, tubos…) con medidas, material y orden de desmontaje. **Por qué no TripoSR o Shap-E:** devuelven una malla única que no se puede desarmar; quedan como alternativa y como comparación.
- **Reglas de seguridad:** "el LLM propone, las reglas disponen".

### 5:00–6:30 · Arquitectura (Rol 5)

Mostrar el diagrama ([arquitectura.md](../arquitectura.md)) y recorrer los dos flujos con el dedo:

1. **Gestos:** cámara → MediaPipe → normalización → reglas → si la confianza llega a 0,8 → visor. Todo en el navegador.
2. **Pedido:** frase → LLM (o reglas) → validación → búsqueda → catálogo, o generación por piezas → visor.

Cerrar con los respaldos: sin backend, catálogo local; sin LLM, reglas; sin cámara, botones y teclado; sin generación, solo catálogo.

### 6:30–11:00 · Demo en vivo (Rol 4)

Ver el paso a paso abajo. Mientras Rol 4 maneja, Rol 2 explica los gestos y Rol 3 los pedidos.

### 11:00–13:00 · Métricas, limitaciones y uso responsable (Rol 5)

- Tabla de métricas con meta y resultado (diapositiva 10). Decir cuáles están pendientes.
- Un hallazgo honesto: "mostrame una silla" devolvía el tornillo. Se corrigió con una regla, no ajustando el umbral, y se midió en un segundo set.
- Limitaciones: iluminación y manos parcialmente ocultas; los modelos generados son aproximados, hechos de formas simples; la interpretación y la generación dependen de internet.
- Uso responsable ([uso_responsable.md](../uso_responsable.md)): el video no sale del equipo; los pedidos escritos sí van a un servicio externo y se avisa en pantalla; lo generado se marca "no apto para fabricación sin revisión"; se bloquean objetos fuera del dominio.

### 13:00–15:00 · Cierre (Rol 1)

1. Retomar la frase de propuesta, igual que al principio.
2. Qué demostramos: con los números de las diapositivas 10 y 12.
3. Valor: Carla ahora lo pide con una frase, lo muestra con la mano y lo desarma frente al cliente para explicar cómo funciona.
4. Qué viene: llevarlo a un display holográfico real o conectarlo al catálogo de productos de una empresa.

**Verificación de coherencia:** la diapositiva 13 tiene que responder las tres preguntas de la diapositiva 2 (qué innovamos, para quién, qué valor agrega).

## Demo paso a paso (4 min 30 s)

Antes de empezar: visor abierto, cámara funcionando, luz frontal, backend con `/health` en `ok` y la bomba de agua ya generada la noche anterior (queda en el catálogo).

| # | Acción | Qué se ve | Si falla |
|---|---|---|---|
| 1 | Mostrar el recuadro de cámara con la mano | Esqueleto de la mano y nombre del gesto; cian si se aplica, ámbar si no alcanza la confianza | Botón **Simular manos**, diciendo que es una simulación |
| 2 | Pellizco y mover; palma abierta y rotar; pellizco con las dos manos y separar; puño | La válvula se mueve, gira, se agranda y vuelve al centro | Flechas, `+` / `−` y `R` |
| 3 | Escribir: **mostrame el motor desde atrás y achicalo un poco** | "Entendí: …", el motor desde atrás y más chico, con la etiqueta "del catálogo" | Elegirlo en la lista del catálogo |
| 4 | Escribir: **una pieza giratoria con esferas de metal** | Aparece el rodamiento: lo encontró por significado, sin la palabra "rodamiento" | Pasar al paso 5 |
| 5 | Escribir: **mostrame la válvula** y hacer **señalar** dos veces | Salen el volante y el vástago; el panel Piezas marca 2 afuera | Botón **Desmontar** o tecla `X` |
| 6 | Tocar una pieza en el modelo | Se resalta y aparece para qué sirve | Tocarla en la lista |
| 7 | Escribir: **mostrame la válvula desarmada** | Todas las piezas separadas | Barra **Vista explotada** |
| 8 | Escribir: **armalo de nuevo** | Las piezas vuelven a su lugar | Botón **Armar todo** |
| 9 | Escribir: **mostrame el despiece de una bomba de agua** | La bomba generada por IA, ya desarmada, con la etiqueta "generado por IA, no apto para fabricación sin revisión" | Si no responde en 5 s, decir que la generación está en caché y pasar al paso 10 |
| 10 | Escribir: **mostrame una pistola** | Mensaje de que está fuera del dominio | — |

**Generación en vivo (opcional, solo si sobra tiempo):** pedir un objeto nuevo, por ejemplo **mostrame una morsa de banco**. El visor muestra "generando con IA… N s". Tarda ⟨pendiente: eval_ensamble⟩ segundos; si pasa de 60, seguir con la presentación y volver al final.

**Si no anda nada:** pasar al video de respaldo (2 a 3 minutos, sección 22) sin pedir disculpas largas: "Les muestro la grabación de la demo de ayer".

## Lo que falta completar

- [ ] Números ⟨pendiente⟩ en el guion y en las diapositivas 6, 7, 10, 12 y 13: latencia de punta a punta, aciertos del LLM, reglas vs. MLP, generación por piezas, pruebas con usuarios.
- [ ] Integrantes y fecha en la diapositiva 1.
- [ ] Tiempos reales del "antes", si se consiguen entrevistas.
- [ ] Nombres en la columna "Quién".
- [ ] Video de respaldo.
- [ ] Dos ensayos cronometrados (objetivo: 13 a 14 minutos) y uno con preguntas.

## Preguntas probables (sección 21)

Cada uno contesta las de su área. Respuestas cortas, basadas en lo que está hecho.

**Negocio**

1. **¿Cuánto cuesta operarlo?** El visor, los gestos y la búsqueda corren en la computadora. Se paga el LLM por uso: interpretar una frase es barato y generar un modelo cuesta ⟨pendiente: eval_ensamble⟩ USD, y se paga una sola vez, porque queda en el catálogo. Hay límites de pedidos por minuto y de generaciones cada 10 minutos.
2. **¿Quién lo compraría?** Empresas que venden equipos o piezas técnicas y hacen demostraciones a clientes.
3. **¿En qué se diferencia de un visor 3D o de realidad aumentada?** No hace falta tener el modelo (se pide con una frase), no hace falta mouse ni celular, y se puede desarmar para explicar cómo funciona por dentro.
4. **¿Cómo mediríamos el retorno?** Tiempo desde que se necesita un modelo hasta que se muestra, cantidad de modelos que se dejan de encargar a diseño, y éxito de tarea de vendedores sin capacitación.

**Técnica**

5. **¿Por qué un MLP si las reglas funcionan?** Las reglas son explicables, pero tienen umbrales fijos que fallan con manos de distinto tamaño u orientación. El MLP aprende de ejemplos de varias personas. Lo medimos con una persona que no se usó para entrenar: ⟨pendiente⟩.
6. **¿Por qué no entrenaron su propio generador 3D?** Hacen falta millones de ejemplos y GPU durante semanas. Usamos modelos preentrenados y, para que el resultado se pueda desarmar, el LLM diseña el objeto por piezas y nosotros lo validamos y lo armamos.
7. **¿Qué pasa si el LLM se equivoca o lo quieren manipular?** Solo puede devolver un JSON con un esquema fijo, que se valida. Se bloquean objetos fuera del dominio, se limitan los pedidos y las medidas, si la confianza es baja se pide confirmación, y todo se puede deshacer. Si el LLM falla, sigue el intérprete por reglas.
8. **¿Funciona con distintas personas y poca luz?** Los puntos se normalizan por el tamaño de la mano. Con poca luz MediaPipe pierde la mano y el visor lo muestra. Las pruebas con personas diversas están en ⟨pendiente⟩.

**Responsabilidad**

9. **¿Y la privacidad de la cámara?** El video se procesa en el navegador y no se guarda ni se envía. Lo único que sale es el texto de los pedidos, y la pantalla lo avisa. El dataset de gestos se grabó con consentimiento y un identificador anónimo.
10. **¿Y si alguien fabrica con un modelo generado?** Cada modelo generado dice "no apto para fabricación sin revisión". Está hecho de formas simples y sirve para mostrar, no para fabricar.

**Alcance y futuro**

11. **¿Qué dejaron afuera?** Los pedidos por voz (solo texto) y la generación con TripoSR o Shap-E como método principal, porque no se puede desarmar. Agregamos el desarmado por piezas, que no estaba en la propuesta original.
12. **¿Cómo lo llevarían a un holograma real?** El software no cambia: se cambia la salida por un display holográfico (por ejemplo, Looking Glass) o una pirámide de Pepper's ghost.
