// Panel "Ajustes" del visor (botón de la barra o tecla K):
//  - Sensibilidad de cada gesto: rotar con las dos palmas, desplazarse con una palma y escalar con dos pellizcos.
//  - Umbral de confianza: qué tan seguro tiene que estar el clasificador para aplicar un gesto.
//  - Inercia al rotar: si el objeto sigue girando al soltar las palmas.
//  - Disposiciones: guardar cómo están acomodados los recuadros (posición, tamaño, qué se ve) y volver a eso.
//  - Modo presentación (tecla P): qué recuadros quedan a la vista además del holograma.
// Todo se recuerda en este navegador.

export const AJUSTES_POR_DEFECTO = {
  rotar: 1, desplazar: 1, escalar: 1, umbral: 0.8, inercia: true,
  presentacion: { camara: false, ia: true },
};

const RANGOS = {
  rotar: [0.3, 3], desplazar: [0.3, 3], escalar: [0.3, 3], umbral: [0.5, 0.95],
};
const MAX_DISPOSICIONES = 8;

/** Ajustes válidos a partir de cualquier cosa guardada (valores fuera de rango o de otro tipo → por defecto). */
export function normalizarAjustes(a) {
  const salida = structuredClone(AJUSTES_POR_DEFECTO);
  if (!a || typeof a !== 'object') return salida;
  for (const [clave, [min, max]] of Object.entries(RANGOS)) {
    const v = Number(a[clave]);
    if (Number.isFinite(v)) salida[clave] = Math.min(max, Math.max(min, v));
  }
  if (typeof a.inercia === 'boolean') salida.inercia = a.inercia;
  for (const clave of ['camara', 'ia']) {
    if (typeof a.presentacion?.[clave] === 'boolean') salida.presentacion[clave] = a.presentacion[clave];
  }
  return salida;
}

const leerJson = (preferencia, clave, porDefecto) => {
  try { return JSON.parse(preferencia(clave, 'null')) ?? porDefecto; } catch { return porDefecto; }
};
const escapar = (t) => String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const veces = (v) => `${v.toFixed(1).replace('.', ',')}×`;

const DESLIZADORES = [
  { clave: 'rotar', nombre: 'Rotar · dos palmas', ayuda: 'Cuánto gira el objeto por movimiento de las manos', formato: veces, paso: 0.1 },
  { clave: 'desplazar', nombre: 'Desplazarse · una palma', ayuda: 'Cuánto se mueve la vista por movimiento de la mano', formato: veces, paso: 0.1 },
  { clave: 'escalar', nombre: 'Escalar · dos pellizcos', ayuda: 'Cuánto zoom da separar o juntar las manos', formato: veces, paso: 0.1 },
  {
    clave: 'umbral', nombre: 'Umbral de confianza', paso: 0.05, formato: (v) => `${Math.round(v * 100)} %`,
    ayuda: 'Más alto: menos gestos aplicados por error, pero tarda más en reconocerlos',
  },
];

/**
 * raiz: la sección del panel. capturar() → disposición actual; aplicar(disposicion); reiniciarDisposicion();
 * alCambiar(ajustes) al mover algo; alPresentacion() entra al modo presentación; alCerrar() al tocar ✕.
 */
export function crearPanelAjustes(raiz, {
  preferencia, recordar, capturar, aplicar, reiniciarDisposicion, alCambiar = () => {}, alPresentacion, alCerrar,
}) {
  let ajustes = normalizarAjustes(leerJson(preferencia, 'ajustes', null));
  let disposiciones = leerJson(preferencia, 'disposiciones', []);
  if (!Array.isArray(disposiciones)) disposiciones = [];

  raiz.innerHTML = `
    <header class="ajustes-cabecera">
      <h2>Ajustes</h2>
      <button type="button" class="ajustes-cerrar" aria-label="Cerrar los ajustes">✕</button>
    </header>
    <section>
      <h3>Sensibilidad de los gestos</h3>
      ${DESLIZADORES.map((d) => `
        <label class="ajuste" title="${escapar(d.ayuda)}">
          <span>${d.nombre}</span><output data-valor="${d.clave}"></output>
          <input type="range" data-ajuste="${d.clave}" min="${RANGOS[d.clave][0]}" max="${RANGOS[d.clave][1]}" step="${d.paso}">
        </label>`).join('')}
      <label class="ajuste-casilla"><input type="checkbox" data-ajuste="inercia"> Inercia al rotar: al soltar las palmas girando, el objeto sigue y frena solo</label>
      <button type="button" class="ajustes-restablecer">Restablecer la sensibilidad</button>
    </section>
    <section>
      <h3>Disposición de la pantalla</h3>
      <p class="ajustes-ayuda">Guarda dónde están la cámara y el recuadro de la red y el LLM, su tamaño y qué se ve.</p>
      <form class="ajustes-guardar">
        <input type="text" maxlength="24" placeholder="Nombre (ej.: Demo)" aria-label="Nombre de la disposición" required>
        <button type="submit">Guardar la actual</button>
      </form>
      <ul class="ajustes-disposiciones"></ul>
      <button type="button" class="ajustes-lugar">Volver todo a su lugar</button>
    </section>
    <section>
      <h3>Modo presentación <kbd>P</kbd></h3>
      <p class="ajustes-ayuda">Oculta la interfaz y deja solo el holograma y lo que elijas. Se sale con P o Esc.</p>
      <label class="ajuste-casilla"><input type="checkbox" data-presentacion="camara"> Mostrar la cámara</label>
      <label class="ajuste-casilla"><input type="checkbox" data-presentacion="ia"> Mostrar la red y el LLM</label>
      <button type="button" class="ajustes-presentar">Entrar al modo presentación</button>
    </section>`;
  const $ = (s) => raiz.querySelector(s);

  function guardar() {
    recordar('ajustes', JSON.stringify(ajustes));
    alCambiar(ajustes);
  }

  function pintar() {
    for (const d of DESLIZADORES) {
      raiz.querySelector(`[data-ajuste="${d.clave}"]`).value = ajustes[d.clave];
      raiz.querySelector(`[data-valor="${d.clave}"]`).textContent = d.formato(ajustes[d.clave]);
    }
    $('[data-ajuste="inercia"]').checked = ajustes.inercia;
    for (const clave of ['camara', 'ia']) raiz.querySelector(`[data-presentacion="${clave}"]`).checked = ajustes.presentacion[clave];
  }

  function pintarDisposiciones() {
    const lista = $('.ajustes-disposiciones');
    lista.innerHTML = disposiciones.length
      ? disposiciones.map((d, i) => `<li><span>${escapar(d.nombre)}</span>`
        + `<button type="button" data-aplicar="${i}">Aplicar</button>`
        + `<button type="button" data-borrar="${i}" aria-label="Borrar «${escapar(d.nombre)}»">✕</button></li>`).join('')
      : '<li class="ajustes-vacio">Todavía no guardaste ninguna.</li>';
  }

  raiz.addEventListener('input', (e) => {
    const clave = e.target.dataset.ajuste;
    if (clave && clave !== 'inercia') {
      ajustes = normalizarAjustes({ ...ajustes, [clave]: Number(e.target.value) });
      pintar();
      guardar();
    }
  });
  raiz.addEventListener('change', (e) => {
    if (e.target.dataset.ajuste === 'inercia') ajustes.inercia = e.target.checked;
    const clave = e.target.dataset.presentacion;
    if (clave) ajustes.presentacion[clave] = e.target.checked;
    if (e.target.dataset.ajuste === 'inercia' || clave) guardar();
  });
  raiz.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.classList.contains('ajustes-cerrar')) alCerrar?.();
    if (b.classList.contains('ajustes-restablecer')) {
      ajustes = normalizarAjustes({ ...AJUSTES_POR_DEFECTO, presentacion: ajustes.presentacion });
      pintar();
      guardar();
    }
    if (b.classList.contains('ajustes-lugar')) reiniciarDisposicion();
    if (b.classList.contains('ajustes-presentar')) alPresentacion?.();
    if (b.dataset.aplicar != null) aplicar(disposiciones[Number(b.dataset.aplicar)]?.datos);
    if (b.dataset.borrar != null) {
      disposiciones.splice(Number(b.dataset.borrar), 1);
      recordar('disposiciones', JSON.stringify(disposiciones));
      pintarDisposiciones();
    }
  });
  $('.ajustes-guardar').addEventListener('submit', (e) => {
    e.preventDefault();
    const entrada = e.target.querySelector('input');
    const nombre = entrada.value.trim();
    if (!nombre) return;
    // Con el mismo nombre, se reemplaza; si no, se agrega (las más viejas se descartan pasado el máximo)
    disposiciones = [{ nombre, datos: capturar() }, ...disposiciones.filter((d) => d.nombre !== nombre)]
      .slice(0, MAX_DISPOSICIONES);
    recordar('disposiciones', JSON.stringify(disposiciones));
    entrada.value = '';
    pintarDisposiciones();
  });

  pintar();
  pintarDisposiciones();
  return { ajustes: () => ajustes };
}
