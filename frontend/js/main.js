// Orquestador: conecta cámara, gestos, visor, catálogo y backend.
import { iniciarManos } from './hands.js';
import { crearOverlay } from './overlay.js';
import { iniciarSimulacion } from './simulador.js';
import { crearClasificador } from './gestures.js';
import { crearControles } from './controles.js';
import { crearVisor } from './viewer.js';
import { cargarCatalogo, buscarLocal } from './catalogo.js';
import { interpretarPedido, pedirObjeto } from './api.js';
import * as ui from './ui.js';

const UMBRAL_CONFIANZA = 0.8;      // sección 7.3, flujo 1
const PASO_ROTACION = Math.PI / 12;
const PASO_ESCALA = 1.15;

const visor = crearVisor(document.getElementById('escena'));
const controles = crearControles(visor);
const clasificar = crearClasificador();
let catalogo = [];
let pedidoActual = 0; // las respuestas de pedidos anteriores se descartan

async function mostrarModelo(url, nombre, origen, id) {
  ui.mostrarEstado('cargando modelo…');
  await visor.cargarModelo(url);
  ui.mostrarOrigen(nombre, origen);
  ui.marcarSeleccionado(id);
  ui.mostrarEstado('listo');
}

// Plan B sin backend: búsqueda por palabras en el catálogo local
async function atenderSinBackend(frase) {
  const encontrado = buscarLocal(frase, catalogo);
  if (encontrado) {
    const { item } = encontrado;
    await mostrarModelo(item.url, item.nombre, item.origen ?? 'catalogo', item.id);
  } else {
    ui.mostrarEstado('No encontré ese objeto en el catálogo. Probá con otro nombre.', 'error');
  }
}

async function ejecutar({ accion, objeto, parametros }, id) {
  const { vista, factor_escala: factor } = parametros;
  switch (accion) {
    case 'mostrar':
      try {
        ui.mostrarEstado('buscando la pieza…');
        // Si tarda, es que no estaba en el catálogo y se está generando (sección 15: indicador de carga)
        const inicio = performance.now();
        const reloj = setInterval(() => {
          const s = Math.round((performance.now() - inicio) / 1000);
          if (s >= 2 && id === pedidoActual) ui.mostrarEstado(`generando con IA… ${s} s`);
        }, 1000);
        const r = await pedirObjeto(objeto).finally(() => clearInterval(reloj));
        if (id !== pedidoActual) return;
        // Un modelo recién generado se suma a la lista del catálogo
        if (r.origen === 'generado') await refrescarCatalogo().catch(console.error);
        await mostrarModelo(r.modelo_url, r.nombre, r.origen, r.id);
      } catch (err) {
        if (id !== pedidoActual) return;
        if (err.estado === 404) return ui.mostrarEstado(err.message, 'error');
        throw err;
      }
      // "mostrame el motor desde atrás" trae también la vista (y quizás la escala)
      if (vista) visor.vista(vista);
      if (factor) visor.escalarA(factor);
      break;
    case 'escalar':
      visor.guardarEstado();
      visor.escalarA(visor.escala() * (factor ?? PASO_ESCALA));
      break;
    case 'vista':
      visor.guardarEstado();
      visor.vista(vista);
      break;
    case 'rotar':
      visor.guardarEstado();
      visor.rotar(Math.PI / 2, 0);
      break;
    case 'reiniciar':
      visor.guardarEstado();
      visor.reiniciar();
      break;
  }
  ui.mostrarEstado('listo');
}

async function atenderPedido(frase) {
  const id = ++pedidoActual;
  ui.mostrarEstado('interpretando…');
  let intencion;
  try {
    intencion = await interpretarPedido(frase);
  } catch (err) {
    if (id !== pedidoActual) return;
    if (err.estado !== undefined) throw err; // el backend respondió con un error (429, 422…)
    console.info('Backend no disponible, se usa el catálogo local:', err.message);
    return atenderSinBackend(frase);
  }

  if (id !== pedidoActual) return;
  ui.mostrarInterpretacion(intencion);
  if (intencion.accion === 'desconocido') {
    const motivo = intencion.motivo ? ` (${intencion.motivo})` : '';
    return ui.mostrarEstado(`No entendí el pedido${motivo}. Probá con "mostrame un engranaje".`, 'error');
  }
  // Sección 16.3: si la confianza es baja, se muestra lo interpretado y se pide confirmación
  if (intencion.requiere_confirmacion) {
    ui.mostrarEstado('esperando confirmación');
    if (!(await ui.pedirConfirmacion())) return ui.mostrarEstado('pedido cancelado');
    if (id !== pedidoActual) return;
  }
  await ejecutar(intencion, id);
}

// --- pedidos por texto y catálogo
document.getElementById('pedido').addEventListener('submit', (e) => {
  e.preventDefault();
  const campo = document.getElementById('frase');
  const frase = campo.value.trim();
  if (!frase) return;
  atenderPedido(frase).catch((err) => {
    console.error(err);
    ui.mostrarEstado(err.estado ? err.message : 'No pude cargar el modelo, ¿probás de nuevo?', 'error');
  });
  campo.select();
});

const elegirDelCatalogo = (item) => mostrarModelo(item.url, item.nombre, item.origen ?? 'catalogo', item.id);

async function refrescarCatalogo() {
  catalogo = await cargarCatalogo();
  ui.mostrarCatalogo(catalogo, elegirDelCatalogo);
}

refrescarCatalogo()
  .then(() => catalogo.length && elegirDelCatalogo(catalogo[0]))
  .catch((err) => {
    console.error(err);
    ui.mostrarEstado('No pude cargar el catálogo', 'error');
  });

// --- control alternativo: botones y teclado (para quien no pueda o no quiera usar gestos)
// Cada acción guarda el estado antes de cambiar el objeto, para poder deshacerla
const conHistorial = (accion) => () => {
  visor.guardarEstado();
  accion();
};

function deshacer() {
  ui.mostrarEstado(visor.deshacer() ? 'cambio deshecho' : 'no hay nada para deshacer');
}

const acciones = {
  izquierda: conHistorial(() => visor.rotar(-PASO_ROTACION, 0)),
  derecha: conHistorial(() => visor.rotar(PASO_ROTACION, 0)),
  arriba: conHistorial(() => visor.rotar(0, PASO_ROTACION)),
  abajo: conHistorial(() => visor.rotar(0, -PASO_ROTACION)),
  agrandar: conHistorial(() => visor.escalarA(visor.escala() * PASO_ESCALA)),
  achicar: conHistorial(() => visor.escalarA(visor.escala() / PASO_ESCALA)),
  reiniciar: conHistorial(() => visor.reiniciar()),
  deshacer,
};

document.getElementById('botones').addEventListener('click', (e) => {
  const accion = e.target.closest('button')?.dataset.accion;
  if (accion) acciones[accion]();
});
document.getElementById('boton-ayuda').addEventListener('click', ui.alternarAyuda);

const TECLAS = {
  ArrowLeft: 'izquierda', ArrowRight: 'derecha', ArrowUp: 'arriba', ArrowDown: 'abajo',
  '+': 'agrandar', '=': 'agrandar', '-': 'achicar', r: 'reiniciar', R: 'reiniciar', z: 'deshacer', Z: 'deshacer',
};
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'd' || e.key === 'D') return ui.alternarDepuracion();
  if (e.key === '?') return ui.alternarAyuda();
  const accion = TECLAS[e.key];
  if (accion) {
    e.preventDefault();
    acciones[accion]();
  }
});

// --- gestos: cámara (o simulación) → clasificador → recuadro de detección → visor
const video = document.getElementById('webcam');
const lienzo = document.getElementById('landmarks');
const overlay = crearOverlay(lienzo);
let detenerSimulacion = null;
let ultimoCuadro = 0;
let fps = 0;

function alDetectar(manos, info) {
  const r = clasificar(manos, info.aspecto);
  // Por debajo del umbral el gesto se muestra pero no se aplica
  const aplicado = r.confianza >= UMBRAL_CONFIANZA ? r.gesto : 'ninguno';
  overlay.dibujar(manos, info.lados ?? [], r, aplicado, { simulacion: info.simulacion });
  ui.mostrarGesto(r.gesto, r.confianza, UMBRAL_CONFIANZA);
  ui.mostrarDepuracion(r, info);
  controles(aplicado, manos);

  const ahora = performance.now();
  if (ultimoCuadro) fps = 0.9 * fps + 0.1 * (1000 / (ahora - ultimoCuadro));
  ultimoCuadro = ahora;
  ui.mostrarFps(fps, info.msProcesamiento);
}

async function encenderCamara() {
  detenerSimulacion?.();
  detenerSimulacion = null;
  overlay.limpiar();
  ui.mostrarCamara('iniciando');
  try {
    const { ancho, alto } = await iniciarManos(video, alDetectar);
    lienzo.width = ancho;
    lienzo.height = alto;
    ui.mostrarCamara('activa');
  } catch (err) {
    console.error(err);
    const motivo = err.name === 'NotAllowedError' ? 'No hay permiso para usar la cámara.'
      : err.name === 'NotFoundError' ? 'No se encontró ninguna cámara.'
      : 'No se pudo iniciar la cámara.';
    ui.mostrarCamara('error', `${motivo} Podés usar los botones o el teclado.`);
  }
}

function simular() {
  lienzo.width = 640;
  lienzo.height = 480;
  ui.mostrarCamara('simulacion');
  detenerSimulacion = iniciarSimulacion(alDetectar);
}

document.getElementById('camara-reintentar').addEventListener('click', encenderCamara);
document.getElementById('camara-simular').addEventListener('click', simular);
document.getElementById('camara-detener').addEventListener('click', () => {
  detenerSimulacion?.();
  detenerSimulacion = null;
  overlay.limpiar();
  controles('ninguno', []);
  ui.mostrarGesto('ninguno', 0, UMBRAL_CONFIANZA);
  ui.mostrarCamara('error', 'Simulación detenida. Podés reintentar con la cámara o simular de nuevo.');
});
document.getElementById('camara-tamano').addEventListener('click', ui.alternarTamanoCamara);

encenderCamara();
