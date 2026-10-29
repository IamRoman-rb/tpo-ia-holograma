// Vista "LLM en vivo" (llm.html): el recorrido completo de cada pedido, de la voz a la acción.
//
// Recibe los pasos de cada pedido del visor por el BroadcastChannel de actividad.js (el mismo que usa el panel de
// actividad) y los dibuja como etapas: entrada (texto o voz con Whisper), los motores que votaron con su
// distribución sobre las acciones y lo que cuenta cada uno (la salida cruda del LLM, las etiquetas BIO del
// transformer, las palabras que más pesaron en Naive Bayes, la probabilidad de cada palabra de ser el objeto), el
// consenso con su fórmula y la decisión bayesiana, la validación y la ejecución. El laboratorio manda una frase
// al backend (/intent) con el motor elegido y muestra lo mismo, sin tocar el visor.
import { escucharDesdeVentana } from './actividad.js';
import { interpretarPedido, salud } from './api.js';
import { nombreAccion, nombreMotor } from './motores.js';

const $ = (id) => document.getElementById(id);
const escapar = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (v) => `${Math.round((v ?? 0) * 100)} %`;
const ms = (v) => (v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1).replace('.', ',')} s` : `${Math.round(v)} ms`);
const num = (v, d = 2) => (v ?? 0).toFixed(d).replace('.', ',');
const MAX_PEDIDOS = 40;

const pedidos = new Map();   // id → { id, frase, porVoz, voz, intencion, objeto, fin, t, laboratorio }
let elegido = null;
let ultimoMensaje = 0;
let siguienteLab = 1;

// ---------------------------------------------------------------- piezas de HTML

function barras(acciones, valores, { max = 5, umbral = null, ganadora = null } = {}) {
  const orden = acciones.map((a, i) => [a, valores[i]]).sort((x, y) => y[1] - x[1]).slice(0, max);
  return `<div class="barras">${orden.map(([a, v]) => `<div class="barra${a === ganadora ? ' ganadora' : ''}">`
    + `<span>${escapar(nombreAccion(a))}</span><div class="riel"><i style="width:${(v * 100).toFixed(1)}%"></i>`
    + `${umbral != null ? `<b style="left:${umbral * 100}%"></b>` : ''}</div><em>${pct(v)}</em></div>`).join('')}</div>`;
}

function palabrasBio(bio) {
  return `<div class="palabras">${bio.map((p) => `<span class="palabra ${p.etiqueta === 'O' ? '' : p.etiqueta}"`
    + ` title="${p.etiqueta} · ${pct(p.p)}">${escapar(p.palabra)}<sup>${p.etiqueta}</sup></span>`).join('')}</div>`;
}

function palabrasCalor(lista) {
  return `<div class="palabras">${lista.map((p) => `<span class="palabra" title="P(objeto) = ${pct(p.p)}"`
    + ` style="background:rgba(61,255,176,${(0.05 + 0.4 * p.p).toFixed(2)});border-color:rgba(61,255,176,${(0.1 + 0.6 * p.p).toFixed(2)})">`
    + `${escapar(p.palabra)}<sup>${Math.round(p.p * 100)}</sup></span>`).join('')}</div>`;
}

function detalleMotor(d) {
  const partes = [];
  if (d.crudo_llm_propio != null) {
    partes.push(`<div><span class="tenue">El LLM escribió${d.tokens_entrada != null ? ` (${d.tokens_entrada} → ${d.tokens_salida} tokens)` : ''}:</span>`
      + `<div class="crudo mono">${escapar(d.crudo_llm_propio)}</div></div>`);
  }
  if (d.bio?.length) partes.push(`<div><span class="tenue">Etiquetas BIO (Viterbi): B empieza el objeto, I sigue</span>${palabrasBio(d.bio)}</div>`);
  if (d.palabras_objeto?.length) partes.push(`<div><span class="tenue">Probabilidad de cada palabra de ser el objeto</span>${palabrasCalor(d.palabras_objeto)}</div>`);
  if (d.rasgos_decisivos?.length) {
    partes.push(`<div><span class="tenue">Palabras que más pesaron</span><div class="palabras">${d.rasgos_decisivos
      .map((r) => `<span class="chip violeta">${escapar(r.replaceAll('_', ' ').replace('<i>', '⟨inicio⟩').replace('<f>', '⟨fin⟩'))}</span>`).join(' ')}</div></div>`);
  }
  return partes.join('');
}

function tarjetaMotor(v, acciones, final) {
  const coincide = v.accion === final;
  return `<article class="motor ${coincide ? 'coincide' : 'difiere'}">
    <header><strong>${escapar(nombreMotor(v.motor))}</strong>
      <span>${v.peso != null ? `<span class="chip" title="peso en el consenso">w = ${num(v.peso)}</span> ` : ''}<span class="tenue">${ms(v.ms)}</span></span></header>
    <div class="salida"><b>${escapar(nombreAccion(v.accion))}</b>${v.objeto ? ` «${escapar(v.objeto)}»` : ''}
      <span class="tenue">· ${pct(v.confianza)}</span></div>
    ${v.distribucion ? barras(acciones, v.distribucion, { max: 3, ganadora: v.accion }) : ''}
    ${detalleMotor(v)}
  </article>`;
}

function etapa(n, titulo, sub, cuerpo) {
  return `<section class="etapa" data-n="${n}"><h3>${titulo}${sub ? ` <small>${sub}</small>` : ''}</h3>${cuerpo}</section>`;
}

// ---------------------------------------------------------------- recorrido de un pedido

function recorrido(p) {
  const i = p.intencion;
  const t = i?.traza ?? {};
  const etapas = [];
  let n = 1;

  // 1. Entrada
  const voz = p.voz
    ? `<span class="chip violeta">voz · ${escapar(p.voz.modelo ?? 'Whisper')}</span> <span class="tenue">confianza ${pct(p.voz.confianza)} · ${ms(p.voz.ms_transcripcion)} · el audio no salió del equipo</span>`
    : p.porVoz ? '<span class="chip violeta">voz · navegador</span>' : `<span class="chip">${p.laboratorio ? 'laboratorio' : 'texto'}</span>`;
  etapas.push(etapa(n++, 'Entrada', p.laboratorio ? 'no se ejecuta en el visor' : '',
    `<div class="frase-grande">«${escapar(p.frase)}»</div><p style="margin-top:6px">${voz}</p>`));
  if (!i) {
    etapas.push(etapa(n++, 'Interpretando…', '', `<p class="tenue">${p.fin ? escapar(p.fin.mensaje ?? '') : 'esperando la respuesta del backend'}</p>`));
    return etapas.join('');
  }

  // 2. Cadena de motores
  const fallas = [t.error && `Claude: ${t.error}`, ...(t.fallas ?? [])].filter(Boolean);
  etapas.push(etapa(n++, 'Quién responde', nombreMotor(i.motor, true),
    `<p>${i.motor === 'llm' ? `Respondió <b>${escapar(t.modelo ?? 'Claude')}</b>${t.tokens_entrada != null ? ` · ${t.tokens_entrada} → ${t.tokens_salida} tokens` : ''} · ${ms(t.ms_llm)}`
      : `Lo resolvió <b>${escapar(nombreMotor(i.motor, true))}</b>${t.ms_consenso != null ? ` en ${ms(t.ms_consenso)}` : ''}.`}</p>`
    + (fallas.length ? `<p class="tenue">Antes: ${fallas.map(escapar).join(' · ')}</p>` : '')));

  // 3. Motores (consenso) o el motor solo
  if (t.votos?.length) {
    etapas.push(etapa(n++, 'Los motores votan', `${t.votos.length} en paralelo`,
      `<div class="motores">${t.votos.map((v) => tarjetaMotor(v, t.acciones, i.accion)).join('')}</div>`));
    // 4. Consenso
    const terminos = t.votos.map((v) => `${num(v.peso)}·log p<sub>${escapar(nombreMotor(v.motor))}</sub>(a)`).join(' + ');
    const P = Math.max(...t.distribucion);
    const costoEjecutar = 1 - P;
    const C = t.costo_pregunta ?? 0.2;
    const pregunta = costoEjecutar > C;
    etapas.push(etapa(n++, 'Consenso', t.pesos_ajustados ? 'pesos ajustados por máxima verosimilitud (L-BFGS)' : 'pesos iguales (sin ajustar)',
      `<div class="formula mono">log P(a | frase) = ${terminos} + b<sub>a</sub> − log Z</div>
       <div style="margin-top:10px">${barras(t.acciones, t.distribucion, { max: 5, ganadora: i.accion })}</div>
       <h3 style="margin:14px 0 8px;font-size:14px">Decisión bayesiana <small class="tenue">conviene preguntar si (1 − P) &gt; C</small></h3>
       <div class="decision">
         <div class="costo ${pregunta ? '' : 'elegido'}"><span class="tenue">Ejecutar: costo esperado = 1 − P</span><strong>${num(costoEjecutar)}</strong></div>
         <div class="costo ${pregunta ? 'elegido' : ''}"><span class="tenue">Pedir confirmación: costo C</span><strong>${num(C, 1)}</strong></div>
       </div>
       <p class="tenue" style="margin-top:8px">${escapar(t.decision ?? '')}</p>`));
  } else {
    const v = { motor: i.motor, accion: i.accion, objeto: i.objeto, confianza: i.confianza, ...t };
    const acciones = t.distribucion ? Object.keys(t.distribucion) : null;
    etapas.push(etapa(n++, 'El motor decide', nombreMotor(i.motor),
      `<div class="motores">${tarjetaMotor({ ...v, distribucion: acciones ? Object.values(t.distribucion) : null }, acciones ?? [], i.accion)}</div>`
      + (!acciones && t.probabilidades ? barras(Object.keys(t.probabilidades), Object.values(t.probabilidades), { ganadora: i.accion }) : '')));
  }

  // 5. Validación
  const validada = { accion: i.accion, objeto: i.objeto, parametros: i.parametros, confianza: i.confianza };
  const avisos = [i.motivo && `<span class="chip error">filtro: ${escapar(i.motivo)}</span>`,
    i.requiere_confirmacion && `<span class="chip alerta">se pide confirmación (confianza ${pct(i.confianza)})</span>`,
    !i.motivo && !i.requiere_confirmacion && '<span class="chip verde">pasa los filtros</span>'].filter(Boolean).join(' ');
  etapas.push(etapa(n++, 'Validación y seguridad', 'esquema, dominio, límites y confianza',
    `<p>${avisos}</p><pre class="json mono">${escapar(JSON.stringify(validada, null, 2))}</pre>`
    + (t.crudo && (t.crudo.accion !== i.accion || (t.crudo.objeto ?? null) !== (i.objeto ?? null))
      ? `<p class="tenue">Antes de los filtros: <span class="mono">${escapar(JSON.stringify(t.crudo))}</span></p>` : '')));

  // 6. Ejecución
  if (!p.laboratorio) {
    const o = p.objeto;
    const objeto = o ? `<p>${o.origen === 'generado' ? '<span class="chip violeta">generado con IA</span>' : '<span class="chip">catálogo</span>'}
      → <b>${escapar(o.nombre)}</b> <span class="tenue">${o.similitud != null ? `similitud ${num(o.similitud)} · ` : ''}${escapar(o.metodo ?? '')} · ${ms(o.tiempo_ms)}</span>
      ${o.correccion?.length ? `<br><span class="tenue">Búsqueda difusa (Levenshtein + BK-tree): ${escapar(o.correccion.join(', '))}</span>` : ''}</p>` : '';
    const fin = p.fin ? `<p>${p.fin.fase === 'error' ? `<span class="chip error">✕ ${escapar(p.fin.mensaje)}</span>` : `<span class="chip verde">✓ ${escapar(p.fin.resumen ?? 'listo')}</span>`}</p>` : '<p class="tenue">en curso…</p>';
    etapas.push(etapa(n++, 'Ejecución en el visor', p.msTotal ? `total ${ms(p.msTotal)}` : '', objeto + fin));
  }
  return etapas.join('');
}

// ---------------------------------------------------------------- historial y selección

function dibujarHistorial() {
  const lista = [...pedidos.values()].reverse();
  $('cantidad').textContent = lista.length ? `${lista.length}` : '';
  if (!lista.length) return;
  $('historial').innerHTML = lista.map((p) => {
    const i = p.intencion;
    const estado = p.fin?.fase === 'error' ? '✕' : p.fin ? '✓' : i ? '…' : '⋯';
    return `<li><button type="button" data-id="${escapar(p.id)}" aria-current="${p.id === elegido}">
      <span class="frase">${p.porVoz ? '🎙 ' : p.laboratorio ? '🧪 ' : ''}${escapar(p.frase)}</span>
      <small>${estado} ${i ? `${escapar(nombreAccion(i.accion))} · ${escapar(nombreMotor(i.motor))} · ${pct(i.confianza)}` : 'interpretando…'}</small>
    </button></li>`;
  }).join('');
}

function mostrar(id) {
  elegido = id;
  const p = pedidos.get(id);
  $('recorrido').innerHTML = p ? recorrido(p) : '';
  $('recorrido-resumen').textContent = p?.intencion ? `${nombreAccion(p.intencion.accion)} · ${pct(p.intencion.confianza)}` : '';
  dibujarHistorial();
}

$('historial').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-id]');
  if (b) mostrar(b.dataset.id);
});

function guardar(p) {
  pedidos.set(p.id, p);
  while (pedidos.size > MAX_PEDIDOS) pedidos.delete(pedidos.keys().next().value);
}

// ---------------------------------------------------------------- mensajes del visor

function recibir(m) {
  if (m.tipo === 'voz') {
    $('escuchando').hidden = m.estado !== 'escuchando';
    $('escuchando').textContent = `🎙 escuchando… ${m.texto ?? ''}`;
    return;
  }
  if (m.tipo !== 'pedido') return;
  const id = `v${m.id}`;
  const p = pedidos.get(id) ?? { id, t: m.t };
  if (m.fase === 'enviado') Object.assign(p, { frase: m.frase, porVoz: m.porVoz, voz: m.voz });
  if (m.fase === 'intencion') Object.assign(p, { intencion: m.intencion, msTotal: m.msTotal });
  if (m.fase === 'objeto_listo') p.objeto = m.respuesta;
  if (m.fase === 'ejecutado' || m.fase === 'error') p.fin = { fase: m.fase, resumen: m.resumen, mensaje: m.mensaje };
  if (m.fase === 'sin_backend') p.fin = { fase: 'error', mensaje: 'backend no disponible: búsqueda local en el catálogo' };
  p.frase ??= '';
  guardar(p);
  // El pedido nuevo se muestra solo si no se está mirando otro a propósito (o se mira el último)
  const ultimoId = [...pedidos.keys()].filter((k) => k.startsWith('v')).at(-2);
  if (!elegido || elegido === id || (m.fase === 'enviado' && (elegido === ultimoId || elegido?.startsWith('v')))) mostrar(id);
  else dibujarHistorial();
}

// ---------------------------------------------------------------- laboratorio

$('laboratorio').addEventListener('submit', async (e) => {
  e.preventDefault();
  const frase = $('lab-frase').value.trim();
  if (!frase) return;
  const id = `l${siguienteLab++}`;
  const p = { id, frase, laboratorio: true, t: Date.now() };
  guardar(p);
  mostrar(id);
  const t0 = performance.now();
  try {
    p.intencion = await interpretarPedido(frase, $('lab-motor').value || null);
    p.msTotal = performance.now() - t0;
  } catch (err) {
    p.fin = { fase: 'error', mensaje: err.message };
  }
  mostrar(id);
});

// ---------------------------------------------------------------- estado

async function estadoMotores() {
  try {
    const s = await salud();
    const m = s.motores ?? {};
    const filas = [['Consenso', m.consenso], ['LLM propio', m.llm_propio], ['Transformer', m.transformer],
      ['Embeddings', m.modelo], ['Naive Bayes', m.bayes], ['Voz', s.voz ? s.voz.replace('whisper ', 'Whisper ') : null]];
    $('estado-motores').innerHTML = filas.map(([k, v]) => `<div class="dato"><span>${k}</span><strong style="font-size:14px">${
      v ? `<span class="chip verde">${escapar(v === true ? 'listo' : v)}</span>` : '<span class="chip error">no instalado</span>'}</strong></div>`).join('');
  } catch {
    $('estado-motores').innerHTML = '<p class="tenue">El backend no responde (uvicorn en el puerto 8000).</p>';
  }
}

setInterval(() => {
  const conectado = Date.now() - ultimoMensaje < 5000;
  $('conexion').classList.toggle('conectada', conectado);
  $('conexion').querySelector('span').textContent = conectado ? 'Conectado al visor' : 'Sin visor abierto (el laboratorio funciona igual)';
}, 1000);

escucharDesdeVentana((m) => { ultimoMensaje = Date.now(); recibir(m); });
estadoMotores();
setInterval(estadoMotores, 30000);

// llm.html?frase=…&motor=…: analiza esa frase al abrir (para mostrar o compartir un ejemplo)
const parametros = new URLSearchParams(location.search);
if (parametros.get('frase')) {
  $('lab-frase').value = parametros.get('frase');
  if (parametros.has('motor')) $('lab-motor').value = parametros.get('motor');
  $('laboratorio').requestSubmit();
}
