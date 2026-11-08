// Recuadro "Red y LLM" del visor: las vistas red.html y llm.html embebidas (con ?embebido, sin su cabecera), en
// tiempo real. Reciben los datos del visor por el mismo BroadcastChannel que usan en otra pestaña (actividad.js).
// Se quita con ✕, se achica y agranda con − y + (tres tamaños) o arrastrando la esquina, y recuerda la pestaña,
// el tamaño y si estaba abierto.
//  - La red solo se carga mientras se ve (redibuja a cada cuadro y no guarda historia que valga la pena).
//  - El LLM queda cargado mientras el recuadro está abierto, para no perder la lista de pedidos al cambiar de pestaña.
export const TAMANOS = ['chico', 'mediano', 'grande'];
const PAGINAS = { red: 'red.html', llm: 'llm.html' };

export function crearRecuadroIA(raiz, { preferencia, recordar, alCambiar = () => {} }) {
  const marcos = Object.fromEntries([...raiz.querySelectorAll('iframe[data-ia]')].map((f) => [f.dataset.ia, f]));
  const pestanas = [...raiz.querySelectorAll('[role="tab"][data-ia]')];
  const aparte = raiz.querySelector('#ia-aparte');
  const boton = (accion) => raiz.querySelector(`[data-ia-accion="${accion}"]`);
  let pestana = preferencia('ia-pestana', 'red') === 'llm' ? 'llm' : 'red';
  let tamano = TAMANOS.includes(preferencia('ia-tamano', 'mediano')) ? preferencia('ia-tamano', 'mediano') : 'mediano';

  const cargar = (nombre) => {
    if (!marcos[nombre].getAttribute('src')) marcos[nombre].src = `${PAGINAS[nombre]}?embebido`;
  };
  const descargar = (nombre) => marcos[nombre].removeAttribute('src');

  function sincronizarMarcos() {
    for (const nombre of Object.keys(marcos)) {
      const activa = nombre === pestana;
      marcos[nombre].hidden = !activa;
      if (raiz.hidden) descargar(nombre);
      else if (activa) cargar(nombre);
      else if (nombre === 'red') descargar(nombre);
    }
  }

  function elegir(nombre) {
    pestana = nombre;
    recordar('ia-pestana', nombre);
    for (const b of pestanas) b.setAttribute('aria-selected', String(b.dataset.ia === nombre));
    aparte.href = PAGINAS[nombre];
    aparte.target = `holograma-${nombre}`;
    sincronizarMarcos();
  }

  function fijarTamano(t) {
    tamano = t;
    recordar('ia-tamano', t);
    raiz.classList.remove(...TAMANOS);
    raiz.classList.add(t);
    raiz.style.width = '';   // descarta el tamaño que se le dio arrastrando la esquina
    raiz.style.height = '';
    const i = TAMANOS.indexOf(t);
    boton('achicar').disabled = i === 0;
    boton('agrandar').disabled = i === TAMANOS.length - 1;
  }

  function mostrar(ver = raiz.hidden) {
    raiz.hidden = !ver;
    recordar('ia-vivo', ver ? 'si' : 'no');
    sincronizarMarcos();
    alCambiar(ver);
  }

  // Si se agrandó arrastrando la esquina, − y + parten del tamaño más cercano al actual
  const paso = (delta) => {
    const i = TAMANOS.indexOf(tamano);
    fijarTamano(TAMANOS[Math.min(TAMANOS.length - 1, Math.max(0, i + delta))]);
  };

  raiz.addEventListener('click', (e) => {
    const tab = e.target.closest('[role="tab"][data-ia]');
    if (tab) return elegir(tab.dataset.ia);
    const accion = e.target.closest('[data-ia-accion]')?.dataset.iaAccion;
    if (accion === 'cerrar') mostrar(false);
    if (accion === 'achicar') paso(-1);
    if (accion === 'agrandar') paso(1);
  });

  fijarTamano(tamano);
  elegir(pestana);
  mostrar(preferencia('ia-vivo', 'si') === 'si');
  return {
    mostrar,
    visible: () => !raiz.hidden,
    /** Lo que se guarda en una disposición: si se ve, qué pestaña, qué tamaño (y el arrastrado a mano). */
    estado: () => ({
      visible: !raiz.hidden, pestana, tamano,
      ancho: raiz.style.width || null, alto: raiz.style.height || null,
    }),
    aplicar(e) {
      if (!e) return;
      if (PAGINAS[e.pestana]) elegir(e.pestana);
      if (TAMANOS.includes(e.tamano)) fijarTamano(e.tamano);
      if (e.ancho) raiz.style.width = e.ancho;
      if (e.alto) raiz.style.height = e.alto;
      if (typeof e.visible === 'boolean' && e.visible === raiz.hidden) mostrar(e.visible);
    },
  };
}
