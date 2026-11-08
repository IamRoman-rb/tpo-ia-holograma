// Etiquetas de las piezas en el modo holograma: el nombre y para qué sirve cada pieza, en dos columnas a los
// costados de la pantalla, con una línea guía hasta la pieza (del color de la pieza). Se actualizan en cada cuadro.
import { distribuirEtiquetas } from './holograma.js';

const SVG = 'http://www.w3.org/2000/svg';
const MARGEN = 16;
const CODO = 26;          // tramo horizontal de la línea guía, al salir de la etiqueta

const ENTRE_SUBCOLUMNAS = 14; // espacio entre dos subcolumnas del mismo lado
// Paneles fijos que las etiquetas no deben tapar: los de arriba bajan el comienzo de la columna, los de abajo
// suben su final. Cada lado se mide aparte (abajo a la derecha está la cámara; arriba a la izquierda, el recuadro de la red y el LLM).
const OBSTACULOS = ['#hud', '#barra', '#ia-vivo', '#origen', '#analisis', '#catalogo', '#camara', '#controles'];

// Medidas según el ancho de la pantalla: en el teléfono, etiquetas angostas y sin descripción
function medidas() {
  const angosta = innerWidth < 760;
  const ancho = angosta ? 104 : 200;
  const H = innerHeight;
  const limites = {};
  for (const lado of ['izquierda', 'derecha']) {
    // Franja de la pantalla que ocupa la columna del borde
    const [x0, x1] = lado === 'izquierda' ? [MARGEN, MARGEN + ancho] : [innerWidth - MARGEN - ancho, innerWidth - MARGEN];
    let arriba = angosta ? 64 : 84;
    let abajo = H - 24;
    for (const sel of OBSTACULOS) {
      const el = document.querySelector(sel);
      if (!el || el.hidden) continue;
      if (el.id === 'camara' && document.body.classList.contains('presentacion')
        && !document.body.classList.contains('pres-camara')) continue; // transparente en el modo presentación
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height || r.right < x0 || r.left > x1) continue;
      if (r.top + r.height / 2 < H / 2) arriba = Math.max(arriba, r.bottom + 20);
      else abajo = Math.min(abajo, r.top - 16);
    }
    limites[lado] = { arriba, abajo };
  }
  return {
    ancho,
    separacion: angosta ? 30 : 48,
    separacionCompacta: angosta ? 18 : 21,
    limites,
    // Subcolumnas hasta donde no lleguen al centro de la pantalla (donde suele estar el objeto), 3 como mucho
    maxSubcolumnas: Math.max(1, Math.min(3, Math.floor((innerWidth / 2 - MARGEN - 60) / (ancho + ENTRE_SUBCOLUMNAS)))),
  };
}

/** alElegir(id): al tocar una etiqueta se elige su pieza. */
export function crearEtiquetas(raiz, { alElegir = () => {} } = {}) {
  const svg = document.createElementNS(SVG, 'svg');
  raiz.append(svg);
  const items = new Map(); // id → { div, linea, punto }
  let activas = true;

  function crear(p) {
    const div = document.createElement('button');
    div.type = 'button';
    div.className = 'etiqueta';
    div.innerHTML = '<strong></strong><span></span>';
    div.addEventListener('click', () => alElegir(p.id));
    raiz.append(div);
    const linea = document.createElementNS(SVG, 'polyline');
    const punto = document.createElementNS(SVG, 'circle');
    punto.setAttribute('r', '3');
    svg.append(linea, punto);
    const item = { div, linea, punto, texto: '' };
    items.set(p.id, item);
    return item;
  }

  function quitar(id) {
    const it = items.get(id);
    it.div.remove();
    it.linea.remove();
    it.punto.remove();
    items.delete(id);
  }

  return {
    /** piezas: la proyección que da el visor en cada cuadro ({ id, nombre, descripcion, color, x, y, opacidad, … }). */
    actualizar(piezas) {
      // Mientras se analiza una pieza, solo se rotula esa
      const visibles = !activas ? [] : piezas.some((p) => p.aislada) ? piezas.filter((p) => p.aislada) : piezas;
      raiz.hidden = visibles.length === 0;
      for (const id of [...items.keys()]) if (!visibles.some((p) => p.id === id)) quitar(id);
      if (!visibles.length) return;

      const { ancho, separacion, separacionCompacta, limites, maxSubcolumnas } = medidas();
      const W = innerWidth;
      const H = innerHeight;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      const lugares = distribuirEtiquetas(
        visibles.map((p) => ({ id: p.id, x: p.x * W, y: p.y * H, prioridad: p.seleccionada || p.aislada })),
        { ancho: W, arriba: 0, abajo: H, separacion, separacionCompacta, limites, maxSubcolumnas });
      // En una ventana muy chica pueden no entrar todas: las que quedaron afuera se sacan
      for (const id of [...items.keys()]) if (!lugares.some((l) => l.id === id)) quitar(id);

      // Cuántas subcolumnas hay de cada lado: las líneas salen por afuera de la más interna
      const subcolumnas = { izquierda: 1, derecha: 1 };
      for (const l of lugares) subcolumnas[l.lado] = Math.max(subcolumnas[l.lado], l.subcolumna + 1);

      for (const l of lugares) {
        const p = visibles.find((q) => q.id === l.id);
        const it = items.get(p.id) ?? crear(p);
        const texto = `${p.nombre}|${p.descripcion}`;
        if (it.texto !== texto) {
          it.texto = texto;
          it.div.querySelector('strong').textContent = p.nombre;
          it.div.querySelector('span').textContent = p.descripcion;
          it.div.title = `${p.nombre}: ${p.descripcion}`;
        }
        const izquierda = l.lado === 'izquierda';
        // Las subcolumnas se agregan hacia el centro de la pantalla
        const corrido = l.subcolumna * (ancho + ENTRE_SUBCOLUMNAS);
        const x = izquierda ? MARGEN + corrido : W - MARGEN - ancho - corrido;
        it.div.style.width = `${ancho}px`;
        it.div.style.transform = `translate(${x}px, ${l.y - (l.compacta ? 10 : 12)}px)`;
        it.div.classList.toggle('derecha', !izquierda);
        it.div.classList.toggle('compacta', l.compacta);
        it.div.classList.toggle('elegida', p.seleccionada);
        it.div.style.setProperty('--color', p.color);
        it.div.style.opacity = String(Math.max(0.15, p.opacidad));

        // Línea guía: sale horizontal de la etiqueta y va derecho al centro de la pieza
        const ax = izquierda ? x + ancho : x;
        // El tramo horizontal cruza las subcolumnas internas por el hueco entre sus etiquetas (van intercaladas)
        const internas = (subcolumnas[l.lado] - 1 - l.subcolumna) * (ancho + ENTRE_SUBCOLUMNAS);
        const codo = ax + (izquierda ? CODO + internas : -CODO - internas);
        const [px, py] = [p.x * W, p.y * H];
        it.linea.setAttribute('points', `${ax},${l.y} ${codo},${l.y} ${px},${py}`);
        it.linea.setAttribute('stroke', p.color);
        it.linea.style.opacity = String(0.55 * p.opacidad);
        it.punto.setAttribute('cx', px);
        it.punto.setAttribute('cy', py);
        it.punto.setAttribute('fill', p.color);
        it.punto.style.opacity = String(p.opacidad);
      }
    },

    fijarActivas(valor) {
      activas = valor;
      if (!activas) for (const id of [...items.keys()]) quitar(id);
      raiz.hidden = !activas || items.size === 0;
    },

    activas: () => activas,
  };
}
