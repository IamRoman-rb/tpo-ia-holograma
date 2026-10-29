// Vista "Red neuronal en vivo" (red.html): cómo la red de gestos convierte una mano en un gesto.
//
// El visor manda, por el BroadcastChannel de actividad.js, los 63 números normalizados de la mano de cada cuadro.
// Acá se pasan por la MISMA red (frontend/ml/mlp_gestos.json, con mlp.js), por el árbol de decisión y por las
// reglas, y se dibuja todo: la mano como entrada, cada capa con sus activaciones, las conexiones que más aportan
// (activación × |peso|), la salida softmax y el umbral. Así se ve funcionar aunque el visor use otro clasificador.
import { escucharDesdeVentana } from './actividad.js';
import { capasLineales, crearClasificadorMlp, URL_MODELO } from './mlp.js';
import { NOMBRES as NOMBRES_RASGOS, URL_ARBOL, crearClasificadorArbol } from './arbol.js';
import { clasificarFotograma } from './gestures.js';

const NOMBRES = {
  pellizco: 'Pellizco', palma_abierta: 'Palma abierta', 'puño': 'Puño', 'señalar': 'Señalar',
  pulgar_arriba: 'Pulgar arriba', pulgar_abajo: 'Pulgar abajo', victoria: 'Victoria', ninguno: 'Ninguno',
  escala_dos_manos: 'Dos manos',
};
const COLORES = {
  pellizco: '#2bd4ff', palma_abierta: '#3dffb0', 'puño': '#ff5a4f', 'señalar': '#ffd23f', pulgar_arriba: '#a77bff',
  pulgar_abajo: '#ff9a3c', victoria: '#ff4fd8', ninguno: '#5f7d8c', escala_dos_manos: '#4f8bff',
};
const ACTIVACION = { relu: 'ReLU', gelu: 'GELU', silu: 'SiLU' };
const METODOS = { mlp: 'Red neuronal', arbol: 'Árbol de decisión', reglas: 'Reglas escritas a mano' };
// Huesos de la mano (los mismos que dibuja MediaPipe)
const HUESOS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const MAX_NODOS = 18;          // neuronas dibujadas por capa (las capas grandes se agrupan)
const ARISTAS_POR_CAPA = 34;   // conexiones dibujadas entre dos capas: las que más aportan
const LARGO_HISTORIA = 240;

const $ = (id) => document.getElementById(id);
const escapar = (t) => String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (v) => `${Math.round((v ?? 0) * 100)} %`;
const fmt = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)).replace('.', ',');

let modelo = null;
let red = null;
let estructura = null;
let arbol = null;
let ultimo = null;          // último cuadro del visor
let salida = null;          // { red, arbol, reglas } calculados para ese cuadro
let ultimoMensaje = 0;
const historia = [];

// ---------------------------------------------------------------- lienzo nítido

function nitido(lienzo, ancho, alto) {
  const ctx = lienzo.getContext('2d');
  const escala = ((lienzo.clientWidth || ancho) / ancho) * (devicePixelRatio || 1);
  const w = Math.round(ancho * escala);
  const h = Math.round(alto * escala);
  if (lienzo.width !== w || lienzo.height !== h) { lienzo.width = w; lienzo.height = h; }
  ctx.setTransform(escala, 0, 0, escala, 0, 0);
  return ctx;
}

// ---------------------------------------------------------------- estructura de la red para dibujarla

// Las capas grandes se agrupan en MAX_NODOS neuronas "representantes" (el promedio de un tramo de neuronas)
function grupos(n, max = MAX_NODOS) {
  const g = Math.min(n, max);
  return Array.from({ length: g }, (_, i) => [Math.floor((i * n) / g), Math.floor(((i + 1) * n) / g)]);
}

function prepararEstructura(m) {
  const capas = capasLineales(m);
  const tamanos = [capas[0].W.length, ...capas.map((c) => c.b.length)];
  const gr = tamanos.map((n, k) => (k === 0 ? Array.from({ length: 21 }, (_, j) => [3 * j, 3 * j + 3]) : grupos(n,
    k === tamanos.length - 1 ? n : MAX_NODOS)));
  // |peso| medio entre cada par de grupos de capas vecinas (fijo: se calcula una vez)
  const pesos = capas.map(({ W }, k) => gr[k].map(([a0, a1]) => gr[k + 1].map(([b0, b1]) => {
    let s = 0;
    for (let a = a0; a < a1; a++) for (let b = b0; b < b1; b++) s += Math.abs(W[a][b]);
    return s / ((a1 - a0) * (b1 - b0));
  })));
  return { tamanos, grupos: gr, pesos, parametros: capas.reduce((s, c) => s + c.W.length * c.b.length + c.b.length, 0) };
}

const mediaAbs = (v, [a, b]) => { let s = 0; for (let i = a; i < b; i++) s += Math.abs(v[i]); return s / (b - a); };

// ---------------------------------------------------------------- dibujo de la red

function dibujarRed(t) {
  const lienzo = $('lienzo-red');
  const W = 1000;
  const H = 512;
  const ctx = nitido(lienzo, W, H);
  ctx.clearRect(0, 0, W, H);
  if (!estructura) {
    ctx.fillStyle = '#8fb3c4';
    ctx.font = '15px system-ui';
    ctx.fillText('Cargando la red…', 24, 40);
    return;
  }
  const { tamanos, grupos: gr, pesos } = estructura;
  const r = salida?.red;
  const act = r?.activaciones;
  const nCapas = tamanos.length;
  // Posiciones: la mano a la izquierda (entrada), las ocultas en el medio, la salida a la derecha
  const xMano = [24, 214];
  const xs = tamanos.map((_, k) => (k === 0 ? 0 : 300 + ((k - 1) * 520) / Math.max(1, nCapas - 2)));
  if (nCapas === 2) xs[1] = 820;
  xs[nCapas - 1] = 820;

  // Mano: los 21 puntos normalizados (origen en la muñeca, unidad = muñeca → nudillo medio)
  const puntos = ultimo?.puntos;
  const posMano = [];
  let cx = 0;
  let cy = 0;
  if (puntos) {
    for (let j = 0; j < 21; j++) { cx += puntos[3 * j] / 21; cy += puntos[3 * j + 1] / 21; }
  }
  const escalaMano = 62;   // píxeles por unidad (muñeca → nudillo medio)
  for (let j = 0; j < 21; j++) {
    const x = puntos ? puntos[3 * j] - cx : 0;
    const y = puntos ? puntos[3 * j + 1] - cy : 0;
    posMano.push([(xMano[0] + xMano[1]) / 2 + x * escalaMano, 262 + y * escalaMano]);
  }
  ctx.fillStyle = 'rgba(143,179,196,0.9)';
  ctx.font = '600 12px system-ui';
  ctx.fillText('ENTRADA · 21 puntos × (x, y, z)', 20, 26);
  ctx.strokeStyle = 'rgba(43,212,255,0.12)';
  ctx.strokeRect(14, 40, 210, 440);

  // Posición vertical de cada neurona dibujada
  const ys = gr.map((g, k) => {
    if (k === 0) return posMano.map((p) => p[1]);
    const n = g.length;
    const alto = Math.min(420, n * 24);
    return g.map((_, i) => 256 - alto / 2 + (n === 1 ? alto / 2 : (i * alto) / (n - 1)));
  });
  const xDe = (k, i) => (k === 0 ? posMano[i][0] : xs[k]);

  // Activación de cada neurona dibujada, normalizada por capa (0 a 1)
  const brillo = gr.map((g, k) => {
    if (!act) return g.map(() => 0);
    const v = k === nCapas - 1 ? r.probabilidades : g.map((rango) => mediaAbs(act[k], rango));
    if (k === nCapas - 1) return v;
    const max = Math.max(1e-6, ...v);
    return v.map((x) => x / max);
  });

  // Conexiones: las que más aportan (activación de la neurona de origen × |peso| medio del tramo)
  ctx.lineCap = 'round';
  for (let k = 0; k < nCapas - 1; k++) {
    const candidatas = [];
    gr[k].forEach((_, i) => gr[k + 1].forEach((__, j) => {
      candidatas.push([brillo[k][i] * pesos[k][i][j], i, j]);
    }));
    candidatas.sort((a, b) => b[0] - a[0]);
    const top = candidatas.slice(0, ARISTAS_POR_CAPA);
    const max = Math.max(1e-9, top[0]?.[0] ?? 0);
    for (const [s, i, j] of top) {
      const a = act ? s / max : 0.08;
      const grad = ctx.createLinearGradient(xDe(k, i), ys[k][i], xDe(k + 1, j), ys[k + 1][j]);
      grad.addColorStop(0, `rgba(43,212,255,${0.05 + 0.55 * a})`);
      grad.addColorStop(1, `rgba(61,255,176,${0.05 + 0.55 * a})`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 0.6 + 2.2 * a;
      ctx.setLineDash(act && a > 0.35 ? [5, 9] : []);
      ctx.lineDashOffset = -t * 0.04;
      ctx.beginPath();
      ctx.moveTo(xDe(k, i), ys[k][i]);
      ctx.lineTo(xDe(k + 1, j), ys[k + 1][j]);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Huesos de la mano
  ctx.strokeStyle = puntos ? 'rgba(230,246,255,0.55)' : 'rgba(143,179,196,0.15)';
  ctx.lineWidth = 2;
  for (const [a, b] of HUESOS) {
    ctx.beginPath();
    ctx.moveTo(...posMano[a]);
    ctx.lineTo(...posMano[b]);
    ctx.stroke();
  }

  // Neuronas
  for (let k = 0; k < nCapas; k++) {
    const salidaFinal = k === nCapas - 1;
    gr[k].forEach((_, i) => {
      const b = brillo[k][i];
      const x = xDe(k, i);
      const y = ys[k][i];
      const radio = k === 0 ? 4.5 : salidaFinal ? 9 : 7;
      const color = salidaFinal ? COLORES[modelo.clases[i]] ?? '#2bd4ff' : '#2bd4ff';
      ctx.shadowColor = color;
      ctx.shadowBlur = act ? 16 * b : 0;
      ctx.fillStyle = salidaFinal ? color : `rgba(43,212,255,${0.12 + 0.88 * b})`;
      ctx.globalAlpha = salidaFinal ? 0.25 + 0.75 * b : 1;
      ctx.beginPath();
      ctx.arc(x, y, radio, 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
      if (!salidaFinal && k > 0) {
        ctx.strokeStyle = 'rgba(43,212,255,0.35)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (salidaFinal) {
        const clase = modelo.clases[i];
        const gana = r && clase === r.gesto;
        ctx.fillStyle = gana ? '#e6f6ff' : 'rgba(143,179,196,0.95)';
        ctx.font = `${gana ? 700 : 400} 13px system-ui`;
        ctx.fillText(`${NOMBRES[clase] ?? clase}`, x + 16, y + 4);
        ctx.font = '12px system-ui';
        ctx.fillStyle = 'rgba(143,179,196,0.9)';
        ctx.fillText(r ? pct(b) : '', x + 120, y + 4);
      }
    });
  }

  // Títulos de las capas
  ctx.font = '600 12px system-ui';
  for (let k = 1; k < nCapas; k++) {
    ctx.fillStyle = 'rgba(143,179,196,0.95)';
    ctx.textAlign = 'center';
    const ultima = k === nCapas - 1;
    const titulo = ultima ? `SALIDA · ${tamanos[k]} clases` : `OCULTA ${k} · ${tamanos[k]}`;
    ctx.fillText(titulo, xs[k] + (ultima ? 40 : 0), 26);
    if (!ultima && act) {
      const activas = act[k].filter((v) => v > 1e-6).length / act[k].length;
      ctx.font = '11.5px system-ui';
      ctx.fillStyle = 'rgba(43,212,255,0.85)';
      ctx.fillText(`${Math.round(activas * 100)} % activas`, xs[k], 42);
      ctx.font = '600 12px system-ui';
    }
    ctx.textAlign = 'left';
  }
  if (r) {
    // El umbral de confianza: si la clase ganadora no lo supera, la mano cuenta como "ninguno"
    ctx.fillStyle = '#ffb84d';
    ctx.font = '12px system-ui';
    ctx.fillText(`umbral ${pct(red.umbral)}`, 820 + 80, 492);
  }
  if (!puntos) {
    ctx.fillStyle = 'rgba(143,179,196,0.9)';
    ctx.font = '14px system-ui';
    ctx.fillText('Sin manos a la vista', 50, 262);
  }
}

// ---------------------------------------------------------------- tarjetas

function dibujarAnillo(p, color) {
  const ctx = nitido($('anillo'), 92, 92);
  ctx.clearRect(0, 0, 92, 92);
  ctx.lineWidth = 9;
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath(); ctx.arc(46, 46, 38, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.shadowColor = color; ctx.shadowBlur = 10;
  ctx.beginPath(); ctx.arc(46, 46, 38, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * p); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#e6f6ff';
  ctx.font = '700 18px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(pct(p), 46, 52);
  ctx.textAlign = 'left';
}

function barras(cont, clases, valores, umbral, ganadora) {
  if (cont.dataset.clases !== clases.join()) {
    cont.dataset.clases = clases.join();
    cont.innerHTML = clases.map((c) => `<div class="barra"><span>${escapar(NOMBRES[c] ?? c)}</span>`
      + '<div class="riel"><i></i><b></b></div><em></em></div>').join('');
  }
  [...cont.children].forEach((fila, k) => {
    const v = valores?.[k] ?? 0;
    fila.querySelector('i').style.width = `${(v * 100).toFixed(1)}%`;
    fila.querySelector('i').style.background = COLORES[clases[k]] ?? 'var(--acento)';
    const b = fila.querySelector('b');
    b.hidden = umbral == null;
    if (umbral != null) b.style.left = `${umbral * 100}%`;
    fila.querySelector('em').textContent = valores ? pct(v) : '—';
    fila.classList.toggle('ganadora', clases[k] === ganadora);
  });
}

function actualizarTarjetas() {
  const r = salida?.red;
  const metodo = ultimo?.simulacion ? 'reglas' : ultimo?.metodo ?? 'mlp';
  if (r) {
    $('gesto').textContent = NOMBRES[r.gesto] ?? r.gesto;
    $('gesto').style.color = COLORES[r.gesto] ?? 'var(--texto)';
    $('gesto-detalle').textContent = `la red le da ${pct(r.probabilidad)}`
      + (ultimo?.demostracion ? ` · mano real de HaGRID etiquetada «${NOMBRES[ultimo.aplicado]}»`
        : `${ultimo?.simulacion ? ' · simulación' : ''} · el visor aplica: ${NOMBRES[ultimo?.aplicado] ?? '—'}`);
    dibujarAnillo(r.probabilidad, COLORES[r.gesto] ?? '#2bd4ff');
  } else {
    $('gesto').textContent = '—';
    $('gesto').style.color = '';
    $('gesto-detalle').textContent = ultimo ? 'Sin manos a la vista' : 'Esperando al visor…';
    dibujarAnillo(0, '#2bd4ff');
  }
  if (modelo) barras($('probabilidades'), modelo.clases, r?.probabilidades, red.umbral, r?.gesto);

  // Comparación de los tres clasificadores
  const filas = [
    ['mlp', 'Red neuronal', r ? `${NOMBRES[r.gesto] ?? r.gesto} · ${pct(r.probabilidad)}` : '—'],
    ['arbol', 'Árbol de decisión', salida?.arbol ? `${NOMBRES[salida.arbol.gesto] ?? salida.arbol.gesto} · ${pct(salida.arbol.probabilidad)}` : (arbol ? '—' : 'sin entrenar')],
    ['reglas', 'Reglas a mano', salida?.reglas ? NOMBRES[salida.reglas] ?? salida.reglas : '—'],
  ];
  $('comparacion').innerHTML = filas.map(([m, nombre, valor]) => `<div class="${m === metodo ? 'activo' : ''}">`
    + `<span>${nombre}${m === metodo ? ' <span class="chip">en uso</span>' : ''}</span><b>${escapar(valor)}</b></div>`).join('');

  // Camino del árbol
  const a = salida?.arbol;
  if (a) {
    $('camino').innerHTML = a.camino.map((c) => `<li><span>${escapar(NOMBRES_RASGOS[c.rasgo] ?? c.rasgo)}</span>`
      + `<span class="mono tenue">${fmt(c.valor)} ${c.menor ? '≤' : '>'} ${fmt(c.umbral)}</span><span></span></li>`).join('')
      + `<li class="hoja"><span><b>${escapar(NOMBRES[a.gesto] ?? a.gesto)}</b></span><span class="mono">${pct(a.probabilidad)}</span><span></span></li>`;
  }
  $('red-resumen').textContent = ultimo ? `${Math.round(ultimo.fps ?? 0)} fps en el visor · ${METODOS[metodo] ?? metodo} en uso` : '';
  actualizarPasos();
}

function dibujarHistoria() {
  const ctx = nitido($('historia'), 800, 250);
  ctx.clearRect(0, 0, 800, 250);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.moveTo(0, 240 - y * 230); ctx.lineTo(800, 240 - y * 230); ctx.stroke(); }
  if (!modelo || !historia.length) return;
  const paso = 800 / LARGO_HISTORIA;
  modelo.clases.forEach((clase, k) => {
    ctx.strokeStyle = COLORES[clase] ?? '#2bd4ff';
    ctx.lineWidth = clase === 'ninguno' ? 1 : 2;
    ctx.beginPath();
    historia.forEach((p, i) => {
      const x = 800 - (historia.length - i) * paso;
      const y = 240 - (p ? p[k] : 0) * 230;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.strokeStyle = 'rgba(255,184,77,0.7)';
  ctx.setLineDash([4, 6]);
  ctx.beginPath(); ctx.moveTo(0, 240 - red.umbral * 230); ctx.lineTo(800, 240 - red.umbral * 230); ctx.stroke();
  ctx.setLineDash([]);
}

function mostrarArquitectura() {
  const capas = estructura.tamanos.join(' → ');
  const g = modelo.genoma;
  const datos = [
    ['Tipo', modelo.version === 3 ? 'Red residual (diseñada por un algoritmo genético)' : 'Perceptrón multicapa'],
    ['Capas', capas],
    ['Parámetros', estructura.parametros.toLocaleString('es-AR')],
    ['Activación', ACTIVACION[modelo.activacion] ?? 'ReLU'],
    ['Umbral de confianza', pct(modelo.umbral_confianza)],
  ];
  if (g) {
    datos.push(['Conexiones residuales', g.residual ? 'sí' : 'no'], ['Normalización por capa', g.normalizacion ? 'LayerNorm' : 'no'],
      ['Dropout', String(g.dropout).replace('.', ',')], ['Tasa de aprendizaje', String(g.tasa)], ['Lote', String(g.lote)],
      ['Aumento de datos', `×${String(g.aumento).replace('.', ',')}`]);
  }
  if (arbol) datos.push(['Árbol de decisión', `profundidad ${arbol.profundidad}`]);
  $('arquitectura').innerHTML = datos.map(([k, v]) => `<div class="dato"><span>${k}</span><strong>${escapar(v)}</strong></div>`).join('');
  $('arquitectura-origen').textContent = [modelo.origen, modelo.fecha].filter(Boolean).join(' · ');
}

function actualizarPasos() {
  if (!estructura) return;
  const r = salida?.red;
  const act = r?.activaciones;
  const vivo = (t) => `<span class="vivo">${t}</span>`;
  const f = ACTIVACION[modelo.activacion] ?? 'ReLU';
  const pasos = [
    ['Detección', `MediaPipe encuentra 21 puntos de cada mano en la imagen de la cámara: 63 números (x, y, z). ${vivo(ultimo ? `${ultimo.manos} mano(s) en el cuadro` : '')}`],
    ['Filtro de Kalman', 'Antes de clasificar, un filtro de Kalman adaptativo saca el temblor de los puntos: estima la posición real como una variable que se mueve con velocidad casi constante.'],
    ['Normalización', 'Se pone el origen en la muñeca y se divide por la distancia muñeca → nudillo medio: el mismo gesto da los mismos números cerca o lejos de la cámara.'],
    ...estructura.tamanos.slice(1, -1).map((n, k) => [`Capa oculta ${k + 1}`, `${n} neuronas calculan z = W·x + b y aplican ${f}`
      + `${modelo.version === 3 && modelo.genoma?.residual ? ' (con conexión residual: se suma la entrada del bloque)' : ''}. `
      + vivo(act ? `${Math.round((act[k + 1].filter((v) => v > 1e-6).length / n) * 100)} % activas ahora` : '')]),
    ['Softmax', `La última capa da un número por gesto; la softmax los convierte en probabilidades que suman 1. ${vivo(r ? `${NOMBRES[r.gesto]}: ${pct(r.probabilidad)}` : '')}`],
    ['Umbral', `Si la probabilidad mayor no llega a ${pct(modelo.umbral_confianza)} (elegido con validación), la mano cuenta como «ninguno»: mejor no hacer nada que hacer algo no pedido.`],
    ['Ventana temporal', `El visor aplica el gesto solo si se repite en los últimos cuadros. ${vivo(ultimo ? `confianza temporal ${pct(ultimo.confianza)}` : '')}`],
  ];
  $('pasos').innerHTML = pasos.map(([t, d]) => `<li><strong>${t}</strong>${d}</li>`).join('');
}

async function mostrarEvolucion() {
  try {
    const r = await fetch('ml/neuroevolucion.json');
    if (!r.ok) return;
    const e = await r.json();
    $('tarjeta-evolucion').hidden = false;
    const ctx = nitido($('evolucion'), 800, 225);
    const evals = e.evaluaciones;
    const gen = evals.filter((v) => v.busqueda === 'genético');
    const azar = evals.filter((v) => v.busqueda === 'al azar');
    const n = Math.max(gen.length, azar.length);
    const min = Math.min(...evals.map((v) => v.f1).filter((v) => v > 0.5), 0.9);
    const y = (f1) => 210 - ((Math.max(f1, min) - min) / (1 - min)) * 195;
    const x = (i) => 30 + (i / Math.max(1, n - 1)) * 760;
    ctx.clearRect(0, 0, 800, 225);
    for (const [lista, color] of [[gen, '#3dffb0'], [azar, '#a77bff']]) {
      ctx.fillStyle = color;
      lista.forEach((v, i) => { ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(x(i), y(v.f1), 2.5, 0, 7); ctx.fill(); });
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      let mejor = 0;
      lista.forEach((v, i) => { mejor = Math.max(mejor, v.f1); if (i) ctx.lineTo(x(i), y(mejor)); else ctx.moveTo(x(i), y(mejor)); });
      ctx.stroke();
    }
    ctx.fillStyle = '#8fb3c4';
    ctx.font = '12px system-ui';
    ctx.fillText(`F1 en validación (desde ${fmt(min)})`, 30, 14);
    ctx.fillText('redes entrenadas →', 690, 222);
    ctx.fillStyle = '#3dffb0'; ctx.fillText('■ algoritmo genético', 250, 14);
    ctx.fillStyle = '#a77bff'; ctx.fillText('■ búsqueda al azar (mismo presupuesto)', 400, 14);
    const el = e.elegida;
    $('evolucion-resumen').textContent = el ? `elegida: ${el.genoma.profundidad} × ${el.genoma.ancho} ${ACTIVACION[el.genoma.activacion]}, F1 ${fmt(el.f1_validacion)}` : '';
  } catch { /* sin curva todavía */ }
}

// ---------------------------------------------------------------- datos

function procesar(m) {
  if (m.tipo !== 'cuadro') return;
  ultimo = m;
  salida = null;
  if (m.puntos && red) {
    const puntos = Array.from({ length: 21 }, (_, j) => ({ x: m.puntos[3 * j], y: m.puntos[3 * j + 1], z: m.puntos[3 * j + 2] }));
    salida = {
      red: red(puntos, 1),
      arbol: arbol ? arbol(puntos, 1) : null,
      reglas: clasificarFotograma(puntos, 1),
    };
  }
  historia.push(salida?.red?.probabilidades ?? null);
  if (historia.length > LARGO_HISTORIA) historia.shift();
}

// Demostración: sin un visor abierto, la vista recorre manos REALES de HaGRID (frontend/ml/manos_demo.json:
// 4 al azar por gesto, de personas de la partición de prueba, que la red no vio), con una transición suave
// entre una y otra. Así la vista se entiende sola, por ejemplo en una exposición.
const demo = new URLSearchParams(location.search).has('demo');   // ?demo: arranca sin esperar al visor
let manosDemo = [];
function cuadroDemo(t) {
  if (!manosDemo.length) return null;
  const fase = (t / 1800) % manosDemo.length;
  const k = Math.floor(fase);
  const mezcla = Math.min(1, Math.max(0, (fase - k - 0.7) / 0.3));   // el último 30 % de cada mano: transición
  const a = manosDemo[k].puntos;
  const b = manosDemo[(k + 1) % manosDemo.length].puntos;
  return { tipo: 'cuadro', demostracion: true, metodo: 'mlp', manos: 1, fps: 30, confianza: 1,
    aplicado: manosDemo[k].clase, puntos: a.map((v, e) => v * (1 - mezcla) + b[e] * mezcla) };
}

let tarjetasPendientes = false;
let ultimaDemo = 0;
function bucle(t) {
  const conectado = Date.now() - ultimoMensaje < 5000;
  if (!conectado && Date.now() - inicio > (demo ? 0 : 2500) && t - ultimaDemo > 33) {
    ultimaDemo = t;
    const cuadro = cuadroDemo(t);
    if (cuadro) { procesar(cuadro); tarjetasPendientes = true; }
  }
  $('conexion').classList.toggle('conectada', conectado);
  $('conexion').querySelector('span').textContent = conectado ? 'Conectado al visor'
    : 'Demostración con manos reales de HaGRID · abrí el visor en otra pestaña para ver la tuya';
  dibujarRed(t);
  if (tarjetasPendientes) {
    tarjetasPendientes = false;
    actualizarTarjetas();
    dibujarHistoria();
  }
  requestAnimationFrame(bucle);
}

const inicio = Date.now();
async function iniciar() {
  try {
    modelo = await (await fetch(URL_MODELO)).json();
    red = crearClasificadorMlp(modelo);
    estructura = prepararEstructura(modelo);
  } catch (err) {
    console.error(err);
  }
  try {
    const datos = await (await fetch(URL_ARBOL)).json();
    arbol = crearClasificadorArbol(datos);
    $('arbol-resumen').textContent = `profundidad ${datos.profundidad} · ${datos.hojas} hojas · aprendido de HaGRID`;
  } catch { $('arbol-resumen').textContent = 'todavía no entrenado (ml/arboles/entrenar_arboles.py)'; }
  if (estructura) mostrarArquitectura();
  fetch('ml/manos_demo.json').then((r) => r.json()).then((d) => { manosDemo = d.manos; }).catch(() => {});
  actualizarTarjetas();
  mostrarEvolucion();
  escucharDesdeVentana((m) => {
    ultimoMensaje = Date.now();
    procesar(m);
    tarjetasPendientes = true;
  });
  requestAnimationFrame(bucle);
}

iniciar();
