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
import * as metricas from './metricas.js';
import { cargarMlp } from './mlp.js';
import { crearVoz, decir, puedeReconocerLocal, reconocimientoDisponible } from './voz.js';

const UMBRAL_CONFIANZA = 0.8;      // sección 7.3, flujo 1
const PASO_ROTACION = Math.PI / 12;
const PASO_ESCALA = 1.15;

const visor = crearVisor(document.getElementById('escena'), {
  alCambiarPiezas: (lista) => {
    ui.mostrarPiezas(lista, (id) => visor.seleccionar(id));
    // El panel de análisis sigue al visor (también si se deshace o se reinicia)
    const aislada = lista.find((p) => p.aislada);
    if (aislada) ui.mostrarAnalisis(visor.infoPieza(aislada.id));
    else ui.ocultarAnalisis();
  },
});

// Abre o cierra el despiece (gesto de victoria sin una pieza elegida)
function alternarDespiece() {
  if (!visor.tienePiezas()) return ui.mostrarEstado('Este modelo es de una sola pieza: no se puede desarmar.', 'error');
  const abierto = visor.hayPiezasAfuera();
  if (abierto) visor.armarTodo();
  else visor.explotar(1);
  ui.fijarExplosion(abierto ? 0 : 1);
  ui.mostrarEstado(abierto ? 'objeto armado' : 'despiece');
}

// Lo que la interfaz muestra cuando un gesto hace algo (controles.js)
const EVENTOS_GESTO = {
  seleccion: (info) => ui.mostrarEstado(`elegida: ${info.nombre}`),
  aislar: (info) => ui.mostrarEstado(`analizando: ${info.nombre}`),
  salir: () => ui.mostrarEstado('listo'),
  despiece: alternarDespiece,
  confirmar: () => { if (!ui.confirmarPendiente()) ui.mostrarEstado('no hay nada para confirmar'); },
  deshacer: (hecho) => ui.mostrarEstado(hecho ? 'cambio deshecho' : 'no hay nada para deshacer'),
};
const controles = crearControles(visor, { alEvento: (tipo, datos) => EVENTOS_GESTO[tipo]?.(datos) });

// Clasificador de gestos: reglas o red neuronal (sección 5.2). Se recuerda la elección en este navegador.
let clasificar = crearClasificador();
const clasificarSimulacion = crearClasificador(); // siempre reglas: ver simulador.js
let mlp = null;
const selectorClasificador = document.getElementById('clasificador');

async function elegirClasificador(metodo) {
  if (metodo === 'mlp') {
    try {
      mlp ??= await cargarMlp();
    } catch (err) {
      console.error(err);
      ui.mostrarEstado('No se pudo cargar la red neuronal: se usan las reglas.', 'error');
      metodo = 'reglas';
    }
  }
  selectorClasificador.value = metodo;
  clasificar = crearClasificador({ mlp: metodo === 'mlp' ? mlp : null });
  try { localStorage.setItem('clasificador', metodo); } catch { /* sin almacenamiento: no se recuerda */ }
}

selectorClasificador.addEventListener('change', () => elegirClasificador(selectorClasificador.value));
let clasificadorGuardado = null;
try { clasificadorGuardado = localStorage.getItem('clasificador'); } catch { /* sin almacenamiento */ }
elegirClasificador(clasificadorGuardado ?? 'mlp');
let catalogo = [];
let pedidoActual = 0; // las respuestas de pedidos anteriores se descartan

let ultimoModelo = { origen: null, t: 0 }; // para medir la latencia del pedido

async function mostrarModelo(url, nombre, origen, id) {
  ui.mostrarEstado('cargando modelo…');
  await visor.cargarModelo(url);
  ultimoModelo = { origen, t: performance.now() };
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

// Busca el objeto (o lo genera) y lo muestra. Devuelve false si no se pudo.
async function cargarObjeto(objeto, id) {
  try {
    ui.mostrarEstado('buscando la pieza…');
    // Si tarda, es que no estaba en el catálogo y se está generando (sección 15: indicador de carga)
    const inicio = performance.now();
    const reloj = setInterval(() => {
      const s = Math.round((performance.now() - inicio) / 1000);
      if (s >= 2 && id === pedidoActual) ui.mostrarEstado(`generando con IA… ${s} s`);
    }, 1000);
    const r = await pedirObjeto(objeto).finally(() => clearInterval(reloj));
    if (id !== pedidoActual) return false;
    // Un modelo recién generado se suma a la lista del catálogo
    if (r.origen === 'generado') await refrescarCatalogo().catch(console.error);
    await mostrarModelo(r.modelo_url, r.nombre, r.origen, r.id);
    return true;
  } catch (err) {
    if (id !== pedidoActual) return false;
    if (err.estado === 404) {
      ui.mostrarEstado(err.message, 'error');
      return false;
    }
    throw err;
  }
}

async function ejecutar({ accion, objeto, parametros }, id) {
  const { vista, factor_escala: factor } = parametros;
  switch (accion) {
    case 'mostrar':
      if (!(await cargarObjeto(objeto, id))) return;
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
      ui.fijarExplosion(0);
      break;
    case 'desmontar':
    case 'explotar':
    case 'armar': {
      // "mostrame la válvula desarmada": primero se carga el objeto
      if (objeto && !(await cargarObjeto(objeto, id))) return;
      if (!visor.tienePiezas()) return ui.mostrarEstado('Este modelo es de una sola pieza: no se puede desarmar.', 'error');
      visor.guardarEstado();
      if (accion === 'desmontar') {
        const nombre = visor.desmontarSiguiente();
        return ui.mostrarEstado(nombre ? `pieza afuera: ${nombre}` : 'no quedan piezas para desmontar');
      }
      if (accion === 'explotar') visor.explotar(1);
      else visor.armarTodo();
      ui.fijarExplosion(accion === 'explotar' ? 1 : 0);
      break;
    }
    case 'aislar': {
      // "mostrame solo el volante": la pieza se busca en el modelo que está en pantalla.
      // Si no es una pieza de este modelo ("mostrame solo un motor"), se busca como objeto.
      const pieza = objeto ? visor.buscarPieza(objeto) : visor.seleccionada();
      if (!pieza && objeto) return ejecutar({ accion: 'mostrar', objeto, parametros }, id);
      if (!visor.tienePiezas()) return ui.mostrarEstado('Este modelo es de una sola pieza.', 'error');
      if (!pieza) return ui.mostrarEstado('Elegí una pieza primero: tocala, o señalala un momento.', 'error');
      visor.guardarEstado();
      visor.aislar(pieza);
      break;
    }
    case 'conjunto':
      visor.salirDeAislamiento();
      break;
  }
  ui.mostrarEstado('listo');
}

async function atenderPedido(frase, { porVoz = false } = {}) {
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
  // Si se pidió hablando, la respuesta también se escucha (sin el porcentaje ni el motor)
  if (porVoz) {
    decir(intencion.accion === 'desconocido' ? 'No entendí el pedido.'
      : document.getElementById('interpretacion').textContent.split(' — ')[0]);
  }
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
let pedidoPorVoz = false;
document.getElementById('pedido').addEventListener('submit', (e) => {
  e.preventDefault();
  const campo = document.getElementById('frase');
  const frase = campo.value.trim();
  const porVoz = pedidoPorVoz;
  pedidoPorVoz = false;
  if (!frase) return;
  const t0 = performance.now();
  atenderPedido(frase, { porVoz }).then(async () => {
    // Latencia de punta a punta: solo cuenta si este pedido terminó mostrando un modelo
    if (ultimoModelo.t < t0) return;
    await metricas.proximoCuadro();
    metricas.registrar(ultimoModelo.origen === 'generado' ? 'pedido_generado' : 'pedido_catalogo', performance.now() - t0);
    ui.mostrarDepuracion(null, null, metricas.lineasResumen());
  }).catch((err) => {
    console.error(err);
    ui.mostrarEstado(err.estado ? err.message : 'No pude cargar el modelo, ¿probás de nuevo?', 'error');
  });
  campo.select();
});

// --- pedidos por voz: el texto reconocido sigue el mismo camino que uno escrito
const botonMicrofono = document.getElementById('microfono');
let voz = null;

async function prepararVoz() {
  const Reconocimiento = reconocimientoDisponible();
  if (!Reconocimiento) return; // Firefox: sin reconocimiento de voz, queda el texto
  const local = await puedeReconocerLocal(Reconocimiento);
  const campo = document.getElementById('frase');
  voz = crearVoz({
    local,
    alParcial: (texto) => { campo.value = texto; },
    alFinal: (texto) => {
      campo.value = texto;
      pedidoPorVoz = true;
      document.getElementById('pedido').requestSubmit();
    },
    alEstado: (estado, detalle) => {
      botonMicrofono.classList.toggle('escuchando', estado === 'escuchando');
      botonMicrofono.setAttribute('aria-pressed', String(estado === 'escuchando'));
      if (estado === 'escuchando') ui.mostrarEstado('escuchando…');
      else if (estado === 'error') ui.mostrarEstado(detalle, 'error');
      else if (detalle) ui.mostrarEstado(detalle);
    },
  });
  botonMicrofono.hidden = false;
  botonMicrofono.title = local ? 'Pedir por voz (V) · se reconoce en este equipo'
    : 'Pedir por voz (V) · el audio lo procesa el servicio de voz del navegador';
}

function alternarVoz() {
  if (!voz) return ui.mostrarEstado('Este navegador no reconoce la voz: escribí el pedido.', 'error');
  if (voz.escuchando) voz.detener();
  else voz.escuchar();
}
botonMicrofono.addEventListener('click', alternarVoz);
prepararVoz().catch(console.error);

const elegirDelCatalogo = (item) => mostrarModelo(item.url, item.nombre, item.origen ?? 'catalogo', item.id);

async function refrescarCatalogo() {
  catalogo = await cargarCatalogo();
  ui.mostrarCatalogo(catalogo, elegirDelCatalogo);
}

// ?modelo=models/evaluacion/<id>.glb abre un modelo de eval/eval_ensamble.py para puntuarlo.
// Solo se aceptan archivos de esa carpeta.
const modeloEvaluacion = new URLSearchParams(location.search).get('modelo');
const MODELO_EVALUACION = /^models\/evaluacion\/[\w-]+\.glb$/;

refrescarCatalogo()
  .then(() => {
    if (modeloEvaluacion && MODELO_EVALUACION.test(modeloEvaluacion)) {
      const id = modeloEvaluacion.split('/').pop().replace('.glb', '');
      return mostrarModelo(modeloEvaluacion, `Evaluación: ${id}`, 'generado', null);
    }
    return catalogo.length && elegirDelCatalogo(catalogo[0]);
  })
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
  reiniciar: conHistorial(() => { visor.reiniciar(); ui.fijarExplosion(0); }),
  deshacer,
  desmontar: conHistorial(() => {
    const nombre = visor.desmontarSiguiente();
    ui.mostrarEstado(nombre ? `pieza afuera: ${nombre}` : 'no quedan piezas para desmontar');
  }),
  montar: conHistorial(() => {
    const nombre = visor.montarUltima();
    ui.mostrarEstado(nombre ? `pieza montada: ${nombre}` : 'el objeto ya está armado');
  }),
  armar: conHistorial(() => { visor.armarTodo(); ui.fijarExplosion(0); }),
  aislar: () => {
    const id = visor.seleccionada();
    if (!id) return ui.mostrarEstado('Elegí una pieza primero: tocala, o señalala un momento.', 'error');
    visor.guardarEstado();
    visor.aislar(id);
  },
  conjunto: () => visor.salirDeAislamiento(),
};
document.getElementById('analisis-volver').addEventListener('click', acciones.conjunto);

// --- panel de piezas: botones, vista explotada y selección con clic sobre el modelo
document.getElementById('piezas').addEventListener('click', (e) => {
  const accion = e.target.closest('button')?.dataset.pieza;
  if (accion) acciones[accion]();
});
const deslizador = document.getElementById('explosion');
deslizador.addEventListener('pointerdown', () => visor.guardarEstado());
deslizador.addEventListener('keydown', () => visor.guardarEstado());
deslizador.addEventListener('input', () => visor.explotar(deslizador.value / 100));

let inicioClic = null;
visor.elemento.addEventListener('pointerdown', (e) => { inicioClic = [e.clientX, e.clientY]; });
visor.elemento.addEventListener('pointerup', (e) => {
  // Un clic (no un arrastre) sobre una pieza la selecciona y muestra qué es
  if (!inicioClic || Math.hypot(e.clientX - inicioClic[0], e.clientY - inicioClic[1]) > 5) return;
  if (visor.tienePiezas()) visor.seleccionar(visor.piezaEn(e.clientX, e.clientY));
});

document.getElementById('botones').addEventListener('click', (e) => {
  const accion = e.target.closest('button')?.dataset.accion;
  if (accion) acciones[accion]();
});
document.getElementById('boton-ayuda').addEventListener('click', ui.alternarAyuda);

const TECLAS = {
  ArrowLeft: 'izquierda', ArrowRight: 'derecha', ArrowUp: 'arriba', ArrowDown: 'abajo',
  '+': 'agrandar', '=': 'agrandar', '-': 'achicar', r: 'reiniciar', R: 'reiniciar', z: 'deshacer', Z: 'deshacer',
  x: 'desmontar', X: 'desmontar', a: 'armar', A: 'armar', i: 'aislar', I: 'aislar', Escape: 'conjunto',
};
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'd' || e.key === 'D') return ui.alternarDepuracion();
  if (e.key === 'v' || e.key === 'V') return alternarVoz();
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
  const t0 = performance.now();
  const r = (info.simulacion ? clasificarSimulacion : clasificar)(manos, info.aspecto);
  // Por debajo del umbral el gesto se muestra pero no se aplica
  const aplicado = r.confianza >= UMBRAL_CONFIANZA ? r.gesto : 'ninguno';
  overlay.dibujar(manos, info.lados ?? [], r, aplicado, { simulacion: info.simulacion });
  ui.mostrarGesto(r.gesto, r.confianza, UMBRAL_CONFIANZA);
  controles(aplicado, manos);
  // Procesamiento del fotograma: detección de MediaPipe + clasificación + dibujo + controles
  metricas.registrar(info.simulacion ? 'fotograma_simulado' : 'fotograma',
    performance.now() - t0 + (info.msProcesamiento ?? 0));
  ui.mostrarDepuracion(r, info, metricas.lineasResumen());

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
