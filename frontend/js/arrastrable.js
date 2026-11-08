// Recuadros que se acomodan arrastrándolos por su cabecera (la cámara y el recuadro de la red y el LLM).
// - Se arrastra desde la cabecera, salvo desde sus botones, pestañas o enlaces.
// - La posición se recuerda en este navegador y siempre queda dentro de la pantalla (también al achicar la
//   ventana o al agrandar el recuadro).
// - Doble clic en la cabecera lo devuelve a su lugar de siempre (el de la hoja de estilos).
// - En el teléfono los recuadros tienen un lugar fijo: ahí no se arrastran.
const MARGEN = 8;                                     // distancia mínima al borde de la pantalla
const INTERACTIVOS = 'button, a, input, select, textarea, [role="tab"]';
const TELEFONO = '(max-width: 760px)';

/** Posición dentro de la pantalla para una caja de ancho × alto (puro, para las pruebas). */
export function dentroDePantalla({ x, y }, { ancho, alto }, { anchoPantalla, altoPantalla }) {
  const limitar = (v, max) => Math.round(Math.min(Math.max(v, MARGEN), Math.max(MARGEN, max - MARGEN)));
  return { x: limitar(x, anchoPantalla - ancho), y: limitar(y, altoPantalla - alto) };
}

export function hacerArrastrable(caja, asa, { clave, preferencia, recordar, alMover = () => {} }) {
  const telefono = matchMedia(TELEFONO);
  let posicion = leer();
  let arrastre = null;   // { id, dx, dy }: puntero y distancia del puntero a la esquina de la caja

  function leer() {
    try {
      const p = JSON.parse(preferencia(clave, 'null'));
      return Number.isFinite(p?.x) && Number.isFinite(p?.y) ? p : null;
    } catch { return null; }
  }

  // Lleva la caja a `posicion` (ajustada a la pantalla), o la deja donde dice la hoja de estilos
  function aplicar() {
    const movida = Boolean(posicion) && !telefono.matches;
    caja.classList.toggle('movido', movida);
    if (!movida) {
      for (const p of ['left', 'top', 'right', 'bottom']) caja.style.removeProperty(p);
      return;
    }
    const r = caja.getBoundingClientRect();
    const p = dentroDePantalla(posicion, { ancho: r.width, alto: r.height },
      { anchoPantalla: innerWidth, altoPantalla: innerHeight });
    Object.assign(caja.style, { left: `${p.x}px`, top: `${p.y}px`, right: 'auto', bottom: 'auto' });
  }

  asa.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || telefono.matches || e.target.closest(INTERACTIVOS)) return;
    const r = caja.getBoundingClientRect();
    arrastre = { id: e.pointerId, dx: e.clientX - r.left, dy: e.clientY - r.top };
    asa.setPointerCapture(e.pointerId);
    document.body.classList.add('arrastrando');
    e.preventDefault();
  });
  asa.addEventListener('pointermove', (e) => {
    if (arrastre?.id !== e.pointerId) return;
    posicion = { x: e.clientX - arrastre.dx, y: e.clientY - arrastre.dy };
    aplicar();
    alMover();
  });
  const soltar = (e) => {
    if (arrastre?.id !== e.pointerId) return;
    arrastre = null;
    document.body.classList.remove('arrastrando');
    // Se guarda donde quedó de verdad (ya ajustada a la pantalla)
    const r = caja.getBoundingClientRect();
    posicion = { x: Math.round(r.left), y: Math.round(r.top) };
    recordar(clave, JSON.stringify(posicion));
  };
  asa.addEventListener('pointerup', soltar);
  asa.addEventListener('pointercancel', soltar);
  asa.addEventListener('dblclick', (e) => {
    if (e.target.closest(INTERACTIVOS)) return;
    posicion = null;
    recordar(clave, 'null');
    aplicar();
    alMover();
  });

  // Si cambia la pantalla o el tamaño del recuadro, que no quede afuera
  addEventListener('resize', aplicar);
  telefono.addEventListener?.('change', aplicar);
  new ResizeObserver(() => { if (posicion && !arrastre) aplicar(); }).observe(caja);
  aplicar();
  return {
    reiniciar: () => { posicion = null; recordar(clave, 'null'); aplicar(); },
    /** Dónde quedó ({ x, y } en píxeles) o null si está en su lugar de siempre (para las disposiciones). */
    posicion: () => (posicion ? { ...posicion } : null),
    fijar(p) {
      posicion = p && Number.isFinite(p.x) && Number.isFinite(p.y) ? { x: p.x, y: p.y } : null;
      recordar(clave, JSON.stringify(posicion));
      aplicar();
      alMover();
    },
  };
}
