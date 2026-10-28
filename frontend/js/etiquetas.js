// Etiquetas de las piezas en el modo holograma: el nombre y para qué sirve cada pieza, en dos columnas a los
// costados de la pantalla, con una línea guía hasta la pieza (del color de la pieza). Se actualizan en cada cuadro.
import { distribuirEtiquetas } from './holograma.js';

const SVG = 'http://www.w3.org/2000/svg';
const MARGEN = 16;
const CODO = 26;          // tramo horizontal de la línea guía, al salir de la etiqueta

// Medidas según el ancho de la pantalla: en el teléfono, etiquetas angostas y sin descripción
function medidas() {
  const angosta = innerWidth < 760;
  // Mientras se analiza una pieza, su panel ocupa la esquina de arriba a la derecha
  const analisis = document.getElementById('analisis');
  const debajoAnalisis = analisis && !analisis.hidden && !angosta ? analisis.getBoundingClientRect().bottom + 24 : 0;
  return {
    ancho: angosta ? 104 : 200,
    separacion: angosta ? 30 : 48,
    arriba: Math.max(angosta ? 64 : 84, debajoAnalisis),
    // Sin la cámara de fondo, abajo a la derecha está el recuadro de la cámara
    abajo: innerHeight - (angosta ? 230 : document.body.classList.contains('holograma-ar') ? 190 : 350),
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

      const { ancho, separacion, arriba, abajo } = medidas();
      const W = innerWidth;
      const H = innerHeight;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      const lugares = distribuirEtiquetas(visibles.map((p) => ({ id: p.id, x: p.x * W, y: p.y * H })),
        { ancho: W, arriba, abajo, separacion });

      for (const l of lugares) {
        const p = visibles.find((q) => q.id === l.id);
        const it = items.get(p.id) ?? crear(p);
        const texto = `${p.nombre}|${p.descripcion}`;
        if (it.texto !== texto) {
          it.texto = texto;
          it.div.querySelector('strong').textContent = p.nombre;
          it.div.querySelector('span').textContent = p.descripcion;
          it.div.title = p.descripcion;
        }
        const izquierda = l.lado === 'izquierda';
        const x = izquierda ? MARGEN : W - MARGEN - ancho;
        it.div.style.width = `${ancho}px`;
        it.div.style.transform = `translate(${x}px, ${l.y - 12}px)`;
        it.div.classList.toggle('derecha', !izquierda);
        it.div.classList.toggle('elegida', p.seleccionada);
        it.div.style.setProperty('--color', p.color);
        it.div.style.opacity = String(Math.max(0.15, p.opacidad));

        // Línea guía: sale horizontal de la etiqueta y va derecho al centro de la pieza
        const ax = izquierda ? MARGEN + ancho : W - MARGEN - ancho;
        const codo = ax + (izquierda ? CODO : -CODO);
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
