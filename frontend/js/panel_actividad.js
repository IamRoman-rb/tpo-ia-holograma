// Panel "Actividad de la IA": muestra en vivo lo que hace cada modelo.
//  - Gestos: las activaciones de cada capa de la red neuronal (entrada 63 → 128 → 64 → 8 clases), las
//    probabilidades de cada clase contra el umbral, y una línea de tiempo con el gesto aplicado. Con las
//    reglas, en lugar de la red se ven las métricas de la mano contra sus umbrales.
//  - LLM: cada pedido, desde que sale hasta que se ejecuta: motor (LLM o reglas), modelo, tokens, tiempos,
//    la respuesta tal como vino y lo que quedó después de los filtros de seguridad, y la búsqueda del objeto.
// Se usa dentro del visor (index.html) y en la ventana aparte (actividad.html). Recibe los mensajes de actividad.js.

import { nombreMotor } from './motores.js';

const NOMBRES = {
  pellizco: 'Pellizco', palma_abierta: 'Palma abierta', 'puño': 'Puño', 'señalar': 'Señalar',
  pulgar_arriba: 'Pulgar arriba', pulgar_abajo: 'Pulgar abajo', victoria: 'Victoria',
  escala_dos_manos: 'Dos manos', ninguno: 'Ninguno',
};
const COLORES = {
  pellizco: '#2bd4ff', palma_abierta: '#3dffb0', 'puño': '#ff5a4f', 'señalar': '#ffd23f',
  pulgar_arriba: '#a77bff', pulgar_abajo: '#ff9a3c', victoria: '#ff4fd8', escala_dos_manos: '#4f8bff',
  ninguno: 'rgba(143, 179, 196, 0.25)',
};
const LARGO_LINEA = 240;   // cuadros en la línea de tiempo (unos 8 s a 30 fps)
const MAX_PEDIDOS = 6;

// Métricas de las reglas que se muestran, con su umbral (ver UMBRALES en gestures.js)
const METRICAS_REGLAS = [
  { clave: 'indice', nombre: 'Índice estirado', umbral: 'dedoExtendido', max: 2 },
  { clave: 'medio', nombre: 'Medio estirado', umbral: 'dedoExtendido', max: 2 },
  { clave: 'anular', nombre: 'Anular estirado', umbral: 'dedoExtendido', max: 2 },
  { clave: 'menique', nombre: 'Meñique estirado', umbral: 'dedoExtendido', max: 2 },
  { clave: 'pulgarIndice', nombre: 'Pulgar ↔ índice', umbral: 'pellizcoEntrar', max: 1.5 },
  { clave: 'pulgarAfuera', nombre: 'Pulgar afuera', umbral: 'pulgarAfuera', max: 1.5 },
  { clave: 'separacionV', nombre: 'Separación en V', umbral: 'separacionV', max: 1.5 },
];

// El lienzo se dibuja siempre en un sistema de ancho x alto, pero con tantos píxeles como ocupa en la pantalla:
// así se ve nítido en el panel chico y en la ventana aparte.
function nitido(lienzo, ctx, ancho, alto) {
  const escala = ((lienzo.clientWidth || ancho) / ancho) * (devicePixelRatio || 1);
  const [w, h] = [Math.round(ancho * escala), Math.round(alto * escala)];
  if (lienzo.width !== w || lienzo.height !== h) {
    lienzo.width = w;
    lienzo.height = h;
  }
  ctx.setTransform(escala, 0, 0, escala, 0, 0);
  return [ancho, alto];
}

const escapar = (t) => String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const ms = (v) => (v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);

export function crearPanelActividad(raiz, { enVentana = false, alAbrirVentana = null, alCerrar = null } = {}) {
  raiz.innerHTML = `
    <header class="act-cabecera">
      <h2>Actividad de la IA</h2>
      ${alAbrirVentana ? '<button type="button" class="act-ventana" title="Abrir en otra ventana (para otra pantalla)">↗ Ventana</button>' : ''}
      ${alCerrar ? '<button type="button" class="act-cerrar" aria-label="Cerrar el panel">✕</button>' : ''}
    </header>
    <section class="act-seccion">
      <h3><span class="act-metodo">Gestos</span><span class="act-estado"></span></h3>
      <canvas class="act-capas" width="380" height="176" aria-label="Activaciones de la red neuronal"></canvas>
      <div class="act-reglas" hidden></div>
      <p class="act-camino" hidden></p>
      <div class="act-probs" role="list"></div>
      <canvas class="act-linea" width="380" height="18" aria-label="Gesto aplicado en los últimos segundos"></canvas>
      <p class="act-nota">Esperando la cámara (o la simulación de manos)…</p>
    </section>
    <section class="act-seccion">
      <h3><span>Pedidos al LLM</span><span class="act-modelo"></span></h3>
      <p class="act-voz" hidden></p>
      <ol class="act-pedidos"><li class="act-vacio">Todavía no hubo pedidos. Escribí o decí uno, por ejemplo "mostrame la válvula desarmada".</li></ol>
    </section>`;
  raiz.classList.toggle('act-en-ventana', enVentana);
  const $ = (s) => raiz.querySelector(s);
  raiz.querySelector('.act-ventana')?.addEventListener('click', alAbrirVentana);
  raiz.querySelector('.act-cerrar')?.addEventListener('click', alCerrar);

  const lienzo = $('.act-capas');
  const ctx = lienzo.getContext('2d');
  const linea = $('.act-linea');
  const ctxLinea = linea.getContext('2d');
  const historia = [];
  const pedidos = new Map(); // id → { li, t0, terminado }
  let ultimo = null;         // último cuadro recibido (se dibuja en el próximo requestAnimationFrame)
  let pendiente = false;
  let filasProbs = [];

  // ---------------------------------------------------------------- gestos

  function armarProbs(clases) {
    const cont = $('.act-probs');
    cont.innerHTML = clases.map((c) => `
      <div class="act-prob" role="listitem" data-clase="${escapar(c)}">
        <span>${escapar(NOMBRES[c] ?? c)}</span>
        <div class="act-barra"><i style="background:${COLORES[c] ?? '#2bd4ff'}"></i><b></b></div>
        <em>—</em>
      </div>`).join('');
    filasProbs = [...cont.children].map((f) => ({ f, i: f.querySelector('i'), b: f.querySelector('b'), em: f.querySelector('em') }));
    cont.dataset.clases = clases.join(',');
  }

  function probs(clases, valores, umbral, ganadora) {
    if ($('.act-probs').dataset.clases !== clases.join(',')) armarProbs(clases);
    filasProbs.forEach(({ f, i, b, em }, k) => {
      const v = valores?.[k] ?? 0;
      i.style.width = `${(v * 100).toFixed(1)}%`;
      b.style.left = umbral != null ? `${umbral * 100}%` : '';
      b.hidden = umbral == null;
      em.textContent = valores ? `${Math.round(v * 100)} %` : '—';
      f.classList.toggle('ganadora', clases[k] === ganadora);
    });
  }

  // Forma de cada capa en el dibujo: la entrada como 21 puntos × xyz, la salida como una columna de clases y
  // cada capa oculta como una grilla de hasta 16 filas, con celdas del tamaño que entre (redes de cualquier tamaño).
  function formas(activaciones, W, H) {
    const ultima = activaciones.length - 1;
    const ocultas = activaciones.slice(1, ultima);
    const fijo = 3 * 7 + 14 + 70;                       // entrada + salida + nombres de las clases
    const porOculta = (W - fijo - 16 * (ocultas.length + 1)) / Math.max(1, ocultas.length);
    return activaciones.map((capa, k) => {
      if (k === 0) return { filas: 21, cols: 3, lado: 7, nombre: 'Entrada' };
      if (k === ultima) return { filas: capa.length, cols: 1, lado: 14, nombre: 'Salida' };
      const filas = Math.min(16, capa.length);
      const cols = Math.ceil(capa.length / filas);
      const lado = Math.max(2, Math.min(9, (H - 24) / filas, porOculta / cols));
      return { filas, cols, lado, nombre: `Oculta ${k} · ${capa.length}` };
    });
  }

  // Cada capa como una grilla de celdas: el brillo es la activación de la neurona (normalizada por capa).
  // Las líneas unen las neuronas más activas de capas vecinas: muestran por dónde "pasa" la señal.
  function dibujarRed(activaciones, probabilidades, clases) {
    const [W, H] = nitido(lienzo, ctx, 380, 176);
    ctx.clearRect(0, 0, W, H);
    const capas = formas(activaciones, W, H);
    const ultima = capas.length - 1;
    const anchos = capas.map((c) => c.cols * c.lado);
    const huecos = (W - anchos.reduce((s, a) => s + a, 0) - 70) / ultima;
    let x = 4;
    const celdas = [];
    activaciones.forEach((capa, k) => {
      const { filas, cols, lado, nombre } = capas[k];
      const alto = filas * lado;
      const y0 = 14 + (H - 14 - alto) / 2;
      const salida = k === ultima;
      const valores = salida ? probabilidades : capa.map(Math.abs);
      const maximo = Math.max(1e-6, ...valores);
      const lista = [];
      valores.forEach((v, n) => {
        // La entrada va por punto de la mano (una fila por punto, x y z en columnas); el resto, por filas
        const fila = k === 0 ? Math.floor(n / 3) : n % filas;
        const col = k === 0 ? n % 3 : Math.floor(n / filas);
        const cx = x + col * lado;
        const cy = y0 + fila * lado;
        const a = salida ? v : v / maximo;
        ctx.fillStyle = salida ? (COLORES[clases[n]] ?? '#2bd4ff') : `rgba(43, 212, 255, ${0.08 + 0.92 * a})`;
        ctx.globalAlpha = salida ? 0.25 + 0.75 * a : 1;
        if (salida) {
          ctx.beginPath();
          ctx.arc(cx + lado / 2, cy + lado / 2, lado / 2 - 1.5, 0, 2 * Math.PI);
          ctx.fill();
        } else {
          ctx.fillRect(cx, cy, Math.max(1, lado - (lado > 4 ? 1.5 : 0.5)), Math.max(1, lado - (lado > 4 ? 1.5 : 0.5)));
        }
        ctx.globalAlpha = 1;
        lista.push({ a, x: cx + lado / 2, y: cy + lado / 2 });
      });
      celdas.push(lista);
      ctx.fillStyle = 'rgba(143, 179, 196, 0.9)';
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(nombre, x, 10);
      x += anchos[k] + huecos;
    });
    // Nombres de las clases al lado de la salida
    const salida = celdas[ultima];
    ctx.font = '10px system-ui, sans-serif';
    salida.forEach((c, n) => {
      ctx.fillStyle = c.a > 0.5 ? '#e6f6ff' : 'rgba(143, 179, 196, 0.8)';
      ctx.fillText(NOMBRES[clases[n]] ?? clases[n], c.x + 10, c.y + 3.5);
    });
    // Conexiones entre las neuronas más activas
    ctx.lineWidth = 1;
    for (let k = 0; k < celdas.length - 1; k++) {
      const top = (lista, n) => [...lista].sort((p, q) => q.a - p.a).slice(0, n);
      const desde = top(celdas[k], 6);
      const hasta = top(celdas[k + 1], k + 1 === ultima ? 2 : 6);
      for (const a of desde) {
        for (const b of hasta) {
          ctx.strokeStyle = `rgba(43, 212, 255, ${0.35 * a.a * b.a})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
  }

  function dibujarReglas(metricas, umbrales) {
    const cont = $('.act-reglas');
    if (!cont.childElementCount) {
      cont.innerHTML = METRICAS_REGLAS.map((m) => `
        <div class="act-prob"><span>${m.nombre}</span><div class="act-barra"><i></i><b></b></div><em></em></div>`).join('');
    }
    [...cont.children].forEach((f, k) => {
      const m = METRICAS_REGLAS[k];
      const v = metricas?.[m.clave];
      f.querySelector('i').style.width = v == null ? '0' : `${Math.min(100, (v / m.max) * 100)}%`;
      f.querySelector('b').style.left = `${(umbrales[m.umbral] / m.max) * 100}%`;
      f.querySelector('em').textContent = v == null ? '—' : v.toFixed(2);
    });
  }

  function dibujarLinea() {
    const [W, H] = nitido(linea, ctxLinea, 380, 18);
    ctxLinea.clearRect(0, 0, W, H);
    const paso = W / LARGO_LINEA;
    historia.forEach((g, i) => {
      ctxLinea.fillStyle = COLORES[g] ?? COLORES.ninguno;
      ctxLinea.fillRect(W - (historia.length - i) * paso, 2, Math.ceil(paso), H - 4);
    });
  }

  function dibujarCuadro(m) {
    const conRed = m.metodo === 'mlp' && !m.simulacion;
    const conArbol = m.metodo === 'arbol' && !m.simulacion;
    $('.act-metodo').textContent = conRed ? 'Gestos · red neuronal (MLP)'
      : conArbol ? `Gestos · árbol de decisión${m.arbol ? ` (profundidad ${m.arbol.profundidad})` : ''}`
        : `Gestos · reglas${m.simulacion ? ' (simulación)' : ''}`;
    $('.act-estado').textContent = `${m.manos} ${m.manos === 1 ? 'mano' : 'manos'} · ${Math.round(m.fps ?? 0)} fps · ${ms(m.ms)}`;
    lienzo.hidden = !conRed;
    $('.act-reglas').hidden = conRed || conArbol;
    $('.act-camino').hidden = !conArbol;
    $('.act-probs').hidden = !(conRed || conArbol); // las reglas no dan probabilidades: se ven sus métricas

    if (conArbol) {
      // El árbol no tiene activaciones: se muestra la cadena de preguntas que llevó a la hoja
      $('.act-camino').textContent = m.arbol ? `Camino: ${m.arbol.texto}` : 'Sin manos a la vista.';
      probs(m.clasesRed ?? [], m.arbol?.probabilidades ?? null, m.arbol?.umbral ?? null, m.porMano?.[0] ?? null);
    } else if (conRed && m.red) {
      dibujarRed(m.red.activaciones, m.red.probabilidades, m.red.clases);
      probs(m.red.clases, m.red.probabilidades, m.red.umbral, m.porMano?.[0]);
    } else if (conRed) {
      const [W, H] = nitido(lienzo, ctx, 380, 176);
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(143, 179, 196, 0.9)';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('Sin manos a la vista: la red está en reposo.', 10, H / 2);
      probs(m.clasesRed ?? [], null, null, null);
    } else {
      dibujarReglas(m.metricas, m.umbralesReglas ?? {});
    }
    const aplicado = m.aplicado !== 'ninguno'
      ? `se aplica <strong>${escapar(NOMBRES[m.aplicado] ?? m.aplicado)}</strong>`
      : m.gesto !== 'ninguno' ? `${escapar(NOMBRES[m.gesto] ?? m.gesto)}, todavía sin confianza suficiente` : 'ningún gesto';
    $('.act-nota').innerHTML = `Confianza temporal ${Math.round(m.confianza * 100)} % (mínimo ${Math.round(m.umbral * 100)} %): ${aplicado}.`;
    dibujarLinea();
  }

  // ---------------------------------------------------------------- pedidos

  function filaPedido(m) {
    let p = pedidos.get(m.id);
    if (p) return p;
    $('.act-vacio')?.remove();
    const li = document.createElement('li');
    li.className = 'act-pedido';
    li.innerHTML = `
      <div class="act-frase">${m.porVoz ? '<span class="act-chip">voz</span>' : ''}«${escapar(m.frase ?? '')}»</div>
      <div class="act-pasos"></div>`;
    $('.act-pedidos').prepend(li);
    p = { li, t0: m.t ?? Date.now(), terminado: false };
    pedidos.set(m.id, p);
    while (pedidos.size > MAX_PEDIDOS) {
      const [viejo] = pedidos.keys();
      pedidos.get(viejo).li.remove();
      pedidos.delete(viejo);
    }
    return p;
  }

  function paso(p, clase, html) {
    const div = document.createElement('div');
    div.className = `act-paso ${clase}`;
    div.innerHTML = html;
    p.li.querySelector('.act-pasos').append(div);
    return div;
  }

  function recibirPedido(m) {
    const p = filaPedido(m);
    switch (m.fase) {
      case 'enviado':
        p.espera = paso(p, 'act-esperando', '<span class="act-girando"></span> interpretando… <span class="act-reloj">0 ms</span>');
        break;
      case 'intencion': {
        p.espera?.remove();
        const i = m.intencion;
        const t = i.traza ?? {};
        const sinLlm = t.error ? `<span class="act-tenue">Claude no disponible: ${escapar(t.error)}</span>` : '';
        const clase = { llm: 'llm', reglas: 'reglas' }[i.motor] ?? 'modelo';
        const motor = i.motor === 'llm'
          ? `<span class="act-chip llm">LLM</span> ${escapar(t.modelo ?? '')}`
          : `<span class="act-chip ${clase}">${escapar(nombreMotor(i.motor))}</span> ${sinLlm}`;
        const tokens = t.tokens_entrada != null ? ` · ${t.tokens_entrada} → ${t.tokens_salida} tokens` : '';
        paso(p, '', `${motor}<br><span class="act-tenue">${t.ms_llm != null ? `LLM ${ms(t.ms_llm)} · ` : ''}backend ${ms(i.tiempo_ms)} · total ${ms(m.msTotal)}${tokens}</span>`);
        // El consenso: qué votó cada motor (el detalle completo está en llm.html)
        if (t.votos?.length) {
          const votos = t.votos.map((v) => `${escapar(nombreMotor(v.motor))} → ${escapar(v.accion)} ${Math.round(v.confianza * 100)} %`).join(' · ');
          paso(p, '', `<span class="act-tenue">Votos: ${votos}</span>`);
        }
        // Los modelos propios dan la probabilidad de cada acción: se ven las tres más altas
        if (t.probabilidades) {
          const lista = Object.entries(t.probabilidades)
            .map(([a, v]) => `${escapar(a)} ${Math.round(v * 100)} %`).join(' · ');
          paso(p, '', `<span class="act-tenue">Probabilidades: ${lista}</span>`);
        }
        const validada = { accion: i.accion, objeto: i.objeto, parametros: i.parametros, confianza: i.confianza };
        const cambio = t.crudo && JSON.stringify({ ...t.crudo, parametros: { vista: null, factor_escala: null, ...t.crudo.parametros } })
          !== JSON.stringify({ ...validada, parametros: { vista: null, factor_escala: null, ...validada.parametros } });
        paso(p, '', `<pre class="act-json">${escapar(JSON.stringify(validada, null, 1).replace(/\n\s*/g, ' '))}</pre>`
          + (cambio ? `<span class="act-tenue">Antes de los filtros: ${escapar(JSON.stringify(t.crudo))}</span>` : ''));
        if (i.motivo) paso(p, 'act-aviso', `Filtro: ${escapar(i.motivo)}`);
        if (i.requiere_confirmacion) paso(p, 'act-aviso', `Confianza ${Math.round(i.confianza * 100)} %: se pide confirmación antes de ejecutar`);
        if (m.modelo) $('.act-modelo').textContent = m.modelo;
        if (i.motor === 'llm' && t.modelo) $('.act-modelo').textContent = t.modelo;
        break;
      }
      case 'sin_backend':
        p.espera?.remove();
        paso(p, 'act-aviso', 'Backend no disponible: búsqueda por palabras en el catálogo local (sin LLM)');
        break;
      case 'objeto':
        p.espera = paso(p, 'act-esperando', `<span class="act-girando"></span> buscando «${escapar(m.consulta)}» en el catálogo… <span class="act-reloj">0 ms</span>`);
        p.t0 = m.t ?? Date.now();
        break;
      case 'objeto_listo': {
        p.espera?.remove();
        const r = m.respuesta;
        const como = r.origen === 'generado'
          ? '<span class="act-chip generado">generado con IA</span>'
          : `<span class="act-chip">catálogo</span> similitud ${r.similitud != null ? r.similitud.toFixed(2) : '—'}${r.metodo ? ` (${escapar(r.metodo)})` : ''}`;
        paso(p, '', `${como} → ${escapar(r.nombre)} <span class="act-tenue">${ms(r.tiempo_ms)}</span>`);
        break;
      }
      case 'ejecutado':
        if (p.terminado) break; // ya se mostró cómo terminó (por ejemplo, un error al buscar el objeto)
        p.terminado = true;
        paso(p, 'act-ok', `✓ ${escapar(m.resumen ?? 'listo')}`);
        break;
      case 'error':
        p.espera?.remove();
        if (p.terminado) break;
        p.terminado = true;
        paso(p, 'act-error', `✕ ${escapar(m.mensaje)}`);
        break;
      default:
        break;
    }
  }

  function recibirVoz(m) {
    const el = $('.act-voz');
    el.hidden = m.estado === 'inactivo' && !m.texto;
    el.innerHTML = m.estado === 'escuchando'
      ? `<span class="act-chip voz">escuchando</span> ${escapar(m.texto || '…')}`
      : m.texto ? `<span class="act-chip">voz</span> ${escapar(m.texto)}` : '';
  }

  // Los cuadros llegan a 30-60 por segundo: se dibuja solo el último, una vez por cuadro de pantalla
  function alPintar() {
    pendiente = false;
    if (ultimo) dibujarCuadro(ultimo);
    for (const p of pedidos.values()) {
      const reloj = p.espera?.querySelector('.act-reloj');
      if (reloj && p.espera.isConnected) reloj.textContent = ms(Date.now() - p.t0);
    }
    if (raiz.querySelector('.act-esperando')) programar();
  }
  function programar() {
    if (!pendiente) {
      pendiente = true;
      requestAnimationFrame(alPintar);
    }
  }

  return {
    recibir(m) {
      if (m.tipo === 'cuadro') {
        historia.push(m.aplicado);
        if (historia.length > LARGO_LINEA) historia.shift();
        ultimo = m;
      } else if (m.tipo === 'pedido') recibirPedido(m);
      else if (m.tipo === 'voz') recibirVoz(m);
      if (!raiz.hidden) programar();
    },
  };
}
