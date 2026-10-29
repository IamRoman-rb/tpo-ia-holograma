// Orquestador: conecta cámara, gestos, visor, catálogo y backend.
import { iniciarManos } from './hands.js';
import { crearOverlay } from './overlay.js';
import { iniciarSimulacion } from './simulador.js';
import { crearClasificador, UMBRALES } from './gestures.js';
import { aVista, crearControles } from './controles.js';
import { crearVisor } from './viewer.js';
import { crearEtiquetas } from './etiquetas.js';
import { camaraAPantalla } from './holograma.js';
import { crearActividad } from './actividad.js';
import { crearPanelActividad } from './panel_actividad.js';
import { cargarCatalogo, buscarLocal } from './catalogo.js';
import { interpretarPedido, pedirObjeto, salud, transcribir } from './api.js';
import { CLASIFICADOR_DE_MOTOR, nombreMotor, respuestaComando } from './motores.js';
import * as ui from './ui.js';
import * as metricas from './metricas.js';
import { cargarMlp } from './mlp.js';
import { cargarArbol, textoCamino } from './arbol.js';
import { crearFiltroManos } from './kalman.js';
import { aVector, normalizar } from './normalize.js';
import { crearVoz, decir, escucharMicrofono, puedeReconocerLocal, reconocimientoDisponible } from './voz.js';

const UMBRAL_CONFIANZA = 0.8;      // sección 7.3, flujo 1
const PASO_ROTACION = Math.PI / 12;
const PASO_ESCALA = 1.15;

// Preferencias de la vista que se recuerdan en este navegador (si hay almacenamiento)
const preferencia = (clave, porDefecto) => {
  try { return localStorage.getItem(clave) ?? porDefecto; } catch { return porDefecto; }
};
const recordar = (clave, valor) => {
  try { localStorage.setItem(clave, valor); } catch { /* sin almacenamiento: no se recuerda */ }
};

// Modo holograma: piezas de colores con aristas y puntos y etiquetas con líneas guía, sobre el fondo oscuro con
// la cámara en su recuadro. Opcional ("Cámara de fondo"): el video a pantalla completa detrás, con las manos
// dibujadas encima (como realidad aumentada).
let modoHolograma = preferencia('holograma', 'si') === 'si';
let camaraDeFondo = preferencia('camara-fondo', 'no') === 'si';
let conEtiquetas = preferencia('etiquetas', 'si') === 'si';
const etiquetas = crearEtiquetas(document.getElementById('etiquetas'), {
  alElegir: (id) => {
    visor.seleccionar(id);
    const info = visor.infoPieza(id);
    if (info) ui.mostrarEstado(`elegida: ${info.nombre}`);
  },
});

const visor = crearVisor(document.getElementById('escena'), {
  alCambiarPiezas: (lista) => {
    ui.mostrarPiezas(lista, (id) => visor.seleccionar(id));
    // El panel de análisis sigue al visor (también si se deshace o se reinicia)
    const aislada = lista.find((p) => p.aislada);
    if (aislada) ui.mostrarAnalisis(visor.infoPieza(aislada.id));
    else ui.ocultarAnalisis();
  },
  alCuadro: ({ piezas }) => etiquetas.actualizar(piezas),
  estilo: modoHolograma ? 'holograma' : 'solido',
});

// Actividad de la IA en tiempo real: panel del visor y ventana aparte (actividad.html)
const actividad = crearActividad();
const raizActividad = document.getElementById('actividad');
const botonActividad = document.querySelector('[data-barra="actividad"]');
function alternarActividad(mostrar = raizActividad.hidden) {
  raizActividad.hidden = !mostrar;
  botonActividad.setAttribute('aria-pressed', String(mostrar));
}
const panelActividad = crearPanelActividad(raizActividad, {
  alAbrirVentana: () => window.open('actividad.html', 'actividad-ia', 'width=460,height=820'),
  alCerrar: () => alternarActividad(false),
});
actividad.escuchar(panelActividad.recibir);
const publicarPedido = (id, fase, datos = {}) => actividad.publicar({ tipo: 'pedido', id, fase, t: Date.now(), ...datos });

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
// Con el video a pantalla completa, el cursor va exactamente donde se ve la punta del dedo
function aPantalla(p) {
  if (!document.body.classList.contains('holograma-ar')) return aVista(p);
  return camaraAPantalla(p, {
    anchoVideo: lienzo.width, altoVideo: lienzo.height, anchoPantalla: innerWidth, altoPantalla: innerHeight,
  });
}
const controles = crearControles(visor, { alEvento: (tipo, datos) => EVENTOS_GESTO[tipo]?.(datos), aPantalla });

// Clasificador de gestos: reglas, red neuronal (sección 5.2) o árbol de decisión. Se recuerda la elección en este
// navegador; también se elige por voz ("usá el árbol para los gestos").
let clasificar = crearClasificador();
const clasificarSimulacion = crearClasificador(); // siempre reglas: ver simulador.js
let mlp = null;
let arbol = null;
let metodoGestos = 'reglas';
const selectorClasificador = document.getElementById('clasificador');
const NOMBRES_CLASIFICADOR = { reglas: 'las reglas', mlp: 'la red neuronal', arbol: 'el árbol de decisión' };

async function elegirClasificador(metodo) {
  try {
    if (metodo === 'mlp') mlp ??= await cargarMlp();
    if (metodo === 'arbol') arbol ??= await cargarArbol();
  } catch (err) {
    console.error(err);
    ui.mostrarEstado(`No se pudo cargar ${NOMBRES_CLASIFICADOR[metodo]}: se usan las reglas.`, 'error');
    metodo = 'reglas';
  }
  selectorClasificador.value = metodo;
  metodoGestos = metodo;
  clasificar = crearClasificador({ modelo: metodo === 'mlp' ? mlp : metodo === 'arbol' ? arbol : null });
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
    publicarPedido(id, 'objeto', { consulta: objeto });
    const r = await pedirObjeto(objeto).finally(() => clearInterval(reloj));
    publicarPedido(id, 'objeto_listo', { respuesta: r });
    if (id !== pedidoActual) return false;
    // Un modelo recién generado se suma a la lista del catálogo
    if (r.origen === 'generado') await refrescarCatalogo().catch(console.error);
    await mostrarModelo(r.modelo_url, r.nombre, r.origen, r.id);
    return true;
  } catch (err) {
    if (id !== pedidoActual) return false;
    if (err.estado === 404) {
      publicarPedido(id, 'error', { mensaje: err.message });
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

// Para el panel de actividad: cómo terminó el pedido, según el último estado que se mostró
function publicarFin(id) {
  const estado = document.getElementById('estado');
  const error = estado.classList.contains('error');
  // Con un error, el estado dice "atención" y el detalle está en el aviso
  const texto = error ? document.getElementById('aviso').textContent || estado.textContent : estado.textContent;
  publicarPedido(id, error ? 'error' : 'ejecutado', { resumen: texto, mensaje: texto });
}

// Motor de los pedidos elegido por voz ("usá el LLM propio"); null = la cadena por defecto del backend
let motorPedidos = null;
try { motorPedidos = localStorage.getItem('motor-pedidos'); } catch { /* sin almacenamiento */ }
const mostrarMotorPedidos = () => {
  document.getElementById('motor-pedidos').textContent = motorPedidos ? nombreMotor(motorPedidos) : 'automático';
};
mostrarMotorPedidos();
let ultimoPedido = null;    // el último pedido atendido (para "¿qué modelo respondió?")
let ultimoGesto = null;     // el último gesto reconocido (para "¿por qué ese gesto?")
let respuestasHabladas = true;
try { respuestasHabladas = localStorage.getItem('respuestas-habladas') !== 'no'; } catch { /* sin almacenamiento */ }

// usar_motor y consultar: no tocan el visor, cambian o explican los modelos. Se responde en pantalla y, si el
// pedido vino por voz, en voz alta.
async function atenderComando(intencion, frase, porVoz) {
  const p = intencion.parametros;
  let aplicado = null;
  if (intencion.accion === 'usar_motor' && p.ambito === 'gestos') {
    await elegirClasificador(CLASIFICADOR_DE_MOTOR[p.motor] ?? 'mlp');
    aplicado = metodoGestos;
  } else if (intencion.accion === 'usar_motor') {
    motorPedidos = p.motor === 'automatico' ? null : p.motor;
    try {
      if (motorPedidos) localStorage.setItem('motor-pedidos', motorPedidos);
      else localStorage.removeItem('motor-pedidos');
    } catch { /* sin almacenamiento */ }
    mostrarMotorPedidos();
  }
  const respuesta = respuestaComando({ ...intencion, frase }, { ultimo: ultimoPedido, gestos: ultimoGesto, aplicado });
  ui.mostrarEstado(respuesta);
  if (porVoz && respuestasHabladas) decir(respuesta);
  return respuesta;
}

async function atenderPedido(frase, { porVoz = false } = {}) {
  const id = ++pedidoActual;
  ui.mostrarEstado('interpretando…');
  const t0 = performance.now();
  // Si vino por voz con Whisper, la transcripción (modelo, confianza, tiempo) también va a la vista del LLM
  publicarPedido(id, 'enviado', { frase, porVoz, voz: porVoz ? ultimaTranscripcion : null });
  let intencion;
  try {
    intencion = await interpretarPedido(frase, motorPedidos);
  } catch (err) {
    if (id !== pedidoActual) return;
    if (err.estado !== undefined) {
      publicarPedido(id, 'error', { mensaje: err.message });
      throw err; // el backend respondió con un error (429, 422…)
    }
    console.info('Backend no disponible, se usa el catálogo local:', err.message);
    publicarPedido(id, 'sin_backend');
    await atenderSinBackend(frase);
    return publicarFin(id);
  }

  publicarPedido(id, 'intencion', { intencion, msTotal: performance.now() - t0 });
  if (id !== pedidoActual) return;
  ui.mostrarInterpretacion(intencion);
  if (intencion.accion === 'usar_motor' || intencion.accion === 'consultar') {
    const respuesta = await atenderComando(intencion, frase, porVoz);
    return publicarPedido(id, 'ejecutado', { resumen: respuesta, mensaje: respuesta });
  }
  if (intencion.accion !== 'desconocido') ultimoPedido = intencion;
  // Si se pidió hablando, la respuesta también se escucha (sin el porcentaje ni el motor)
  if (porVoz && respuestasHabladas) {
    decir(intencion.accion === 'desconocido' ? 'No entendí el pedido.'
      : document.getElementById('interpretacion').textContent.split(' — ')[0]);
  }
  if (intencion.accion === 'desconocido') {
    const motivo = intencion.motivo ? ` (${intencion.motivo})` : '';
    ui.mostrarEstado(`No entendí el pedido${motivo}. Probá con "mostrame un engranaje".`, 'error');
    return publicarFin(id);
  }
  // Sección 16.3: si la confianza es baja, se muestra lo interpretado y se pide confirmación
  if (intencion.requiere_confirmacion) {
    ui.mostrarEstado('esperando confirmación');
    if (!(await ui.pedirConfirmacion())) {
      ui.mostrarEstado('pedido cancelado');
      return publicarPedido(id, 'error', { mensaje: 'cancelado por el usuario' });
    }
    if (id !== pedidoActual) return;
  }
  await ejecutar(intencion, id);
  if (id === pedidoActual) publicarFin(id);
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
    publicarFin(pedidoActual);
  });
  campo.select();
});

// --- pedidos por voz: el texto reconocido sigue el mismo camino que uno escrito
// Con el backend, la voz la reconoce Whisper en este equipo (backend/voz.py); si no, el navegador (Web Speech).
const botonMicrofono = document.getElementById('microfono');
const botonManosLibres = document.getElementById('manos-libres');
let voz = null;            // Web Speech (respaldo)
let vozLocal = false;      // ¿hay Whisper en el backend?
let microfono = null;      // grabación en curso (Whisper)
let manosLibres = null;    // escucha continua (Whisper)

function mostrarEscucha(estado) {
  const escuchando = estado === 'escuchando' || estado === 'hablando';
  actividad.publicar({ tipo: 'voz', estado: escuchando ? 'escuchando' : 'inactivo', texto: '' });
  botonMicrofono.classList.toggle('escuchando', escuchando && !manosLibres);
  botonMicrofono.setAttribute('aria-pressed', String(escuchando && !manosLibres));
}

function pedirPorVoz(texto) {
  const campo = document.getElementById('frase');
  campo.value = texto;
  actividad.publicar({ tipo: 'voz', estado: 'final', texto });
  pedidoPorVoz = true;
  document.getElementById('pedido').requestSubmit();
}

let ultimaTranscripcion = null;
async function reconocerFrase(audio, { activacion = false } = {}) {
  ui.mostrarEstado('reconociendo la voz…');
  try {
    const r = await transcribir(audio, { activacion });
    ultimaTranscripcion = r;
    metricas.registrar('voz_whisper', r.ms_transcripcion);
    return r;
  } catch (err) {
    ui.mostrarEstado(`No pude reconocer la voz: ${err.message}`, 'error');
    return null;
  }
}

const sinMicrofono = (err) => ui.mostrarEstado(err.name === 'NotAllowedError'
  ? 'No hay permiso para usar el micrófono.' : 'No pude abrir el micrófono.', 'error');

async function prepararVoz() {
  const campo = document.getElementById('frase');
  const privacidad = document.getElementById('privacidad-voz');
  try {
    vozLocal = Boolean((await salud()).voz) && Boolean(globalThis.MediaRecorder);
  } catch { vozLocal = false; }
  if (vozLocal) {
    botonMicrofono.hidden = false;
    botonManosLibres.hidden = false;
    botonMicrofono.title = 'Pedir por voz (V) · Whisper la reconoce en este equipo';
    privacidad.textContent = 'Los pedidos por voz los reconoce Whisper en este equipo: el audio no sale de la '
      + 'computadora y no se guarda. Solo se escucha mientras el micrófono o el modo manos libres están encendidos.';
    return;
  }
  const Reconocimiento = reconocimientoDisponible();
  if (!Reconocimiento) return; // Firefox sin backend: sin reconocimiento de voz, queda el texto
  const local = await puedeReconocerLocal(Reconocimiento);
  voz = crearVoz({
    local,
    alParcial: (texto) => {
      campo.value = texto;
      actividad.publicar({ tipo: 'voz', estado: 'escuchando', texto });
    },
    alFinal: pedirPorVoz,
    alEstado: (estado, detalle) => {
      // Lo reconocido queda en el pedido (con la marca "voz"); acá solo se muestra mientras se escucha
      mostrarEscucha(estado);
      if (estado === 'escuchando') ui.mostrarEstado('escuchando…');
      else if (estado === 'error') ui.mostrarEstado(detalle, 'error');
      else if (detalle) ui.mostrarEstado(detalle);
    },
  });
  botonMicrofono.hidden = false;
  botonMicrofono.title = local ? 'Pedir por voz (V) · se reconoce en este equipo'
    : 'Pedir por voz (V) · el audio lo procesa el servicio de voz del navegador';
}

async function alternarVoz() {
  if (vozLocal) {
    if (manosLibres) return ui.mostrarEstado('El modo manos libres ya está escuchando: decí "holograma, …".');
    if (microfono) return microfono.detener();
    try {
      microfono = await escucharMicrofono({
        alEstado: (estado) => {
          mostrarEscucha(estado);
          if (estado === 'escuchando') ui.mostrarEstado('escuchando…');
          if (estado === 'hablando') ui.mostrarEstado('te escucho…');
          if (estado === 'listo') microfono = null;
        },
        alFrase: async (audio) => {
          const r = await reconocerFrase(audio);
          if (r?.texto) pedirPorVoz(r.texto);
          else if (r) ui.mostrarEstado('No escuché nada.');
        },
      });
    } catch (err) {
      microfono = null;
      sinMicrofono(err);
    }
    return;
  }
  if (!voz) return ui.mostrarEstado('Este navegador no reconoce la voz: escribí el pedido.', 'error');
  if (voz.escuchando) voz.detener();
  else voz.escuchar();
}

// Manos libres: escucha sin parar y atiende solo lo que empieza con "holograma". Si se dice solo "holograma",
// el sistema contesta "¿sí?" y la frase siguiente no necesita la palabra.
let esperandoOrden = 0;
async function alternarManosLibres() {
  if (!vozLocal) return ui.mostrarEstado('El modo manos libres necesita el backend con Whisper.', 'error');
  if (manosLibres) {
    manosLibres.detener();
    manosLibres = null;
    botonManosLibres.classList.remove('escuchando');
    botonManosLibres.setAttribute('aria-pressed', 'false');
    return ui.mostrarEstado('manos libres apagado');
  }
  microfono?.detener();
  try {
    manosLibres = await escucharMicrofono({
      continuo: true,
      alEstado: (estado) => {
        if (estado === 'hablando') ui.mostrarEstado('te escucho…');
        else if (estado === 'escuchando') ui.mostrarEstado('manos libres: decí "holograma, …"');
      },
      alFrase: async (audio) => {
        const sinPalabra = performance.now() < esperandoOrden;
        const r = await reconocerFrase(audio, { activacion: !sinPalabra });
        if (!r || r.sin_voz) return;
        if (sinPalabra && r.texto) { esperandoOrden = 0; return pedirPorVoz(r.texto); }
        if (r.activado && r.texto) return pedirPorVoz(r.texto);
        if (r.activado) {
          esperandoOrden = performance.now() + 6000;
          if (respuestasHabladas) decir('¿Sí?');
          return ui.mostrarEstado('¿sí? te escucho');
        }
        ui.mostrarEstado(`escuché «${r.texto}», pero no empezó con "holograma"`);
      },
    });
    botonManosLibres.classList.add('escuchando');
    botonManosLibres.setAttribute('aria-pressed', 'true');
    if (respuestasHabladas) decir('Manos libres activado. Decí holograma y tu pedido.');
  } catch (err) {
    manosLibres = null;
    sinMicrofono(err);
  }
}
botonManosLibres.addEventListener('click', alternarManosLibres);

function alternarRespuestasHabladas() {
  respuestasHabladas = !respuestasHabladas;
  try { localStorage.setItem('respuestas-habladas', respuestasHabladas ? 'si' : 'no'); } catch { /* sin almacenamiento */ }
  ui.mostrarEstado(respuestasHabladas ? 'respuestas habladas encendidas' : 'respuestas habladas apagadas');
  if (respuestasHabladas) decir('Respuestas habladas encendidas.');
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
  if (e.key === 'm' || e.key === 'M') return alternarManosLibres();
  if (e.key === 'j' || e.key === 'J') return alternarRespuestasHabladas();
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
// Filtro de Kalman sobre los puntos de la cámara: saca el temblor de MediaPipe antes de clasificar y de mover
// el objeto (la simulación ya trae su propio ruido controlado y no se filtra)
const filtroManos = crearFiltroManos();

function alDetectar(manosDetectadas, info) {
  const t0 = performance.now();
  const manos = info.simulacion ? manosDetectadas : filtroManos.filtrar(manosDetectadas, info.lados ?? [], t0);
  const r = (info.simulacion ? clasificarSimulacion : clasificar)(manos, info.aspecto);
  // Por debajo del umbral el gesto se muestra pero no se aplica
  const aplicado = r.confianza >= UMBRAL_CONFIANZA ? r.gesto : 'ninguno';
  overlay.dibujar(manos, info.lados ?? [], r, aplicado, {
    simulacion: info.simulacion, holograma: document.body.classList.contains('holograma-ar'), zoom: visor.escala(),
  });
  ui.mostrarGesto(r.gesto, r.confianza, UMBRAL_CONFIANZA);
  controles(aplicado, manos);
  // Procesamiento del fotograma: detección de MediaPipe + clasificación + dibujo + controles
  const msCuadro = performance.now() - t0 + (info.msProcesamiento ?? 0);
  metricas.registrar(info.simulacion ? 'fotograma_simulado' : 'fotograma', msCuadro);
  // Panel de actividad: qué vio el clasificador en este cuadro (con el MLP, también sus activaciones)
  const red = r.red?.[0];
  if (manos.length && r.gesto !== 'ninguno' && !info.simulacion) {
    ultimoGesto = { metodo: metodoGestos, gesto: r.gesto, confianza: r.confianza,
      camino: red?.camino ? textoCamino(red.camino, r.porMano?.[0] ?? r.gesto) : null };
  }
  actividad.publicar({
    tipo: 'cuadro', metodo: info.simulacion ? 'reglas' : metodoGestos, simulacion: Boolean(info.simulacion),
    // Los puntos de la primera mano, normalizados: red.html los pasa por la red y el árbol para dibujarlos
    puntos: manos[0] ? aVector(normalizar(manos[0], info.aspecto)).map((v) => Math.round(v * 1000) / 1000) : null,
    gesto: r.gesto, confianza: r.confianza, aplicado, umbral: UMBRAL_CONFIANZA, porMano: r.porMano ?? [],
    manos: manos.length, ms: msCuadro, fps, metricas: r.metricas?.[0] ?? null, umbralesReglas: UMBRALES,
    clasesRed: (metodoGestos === 'arbol' ? arbol : mlp)?.clases,
    red: red && metodoGestos === 'mlp'
      ? { clases: mlp.clases, umbral: mlp.umbral, probabilidades: red.probabilidades, activaciones: red.activaciones } : null,
    arbol: red && metodoGestos === 'arbol'
      ? { clases: arbol.clases, umbral: arbol.umbral, probabilidades: red.probabilidades, camino: red.camino,
        texto: textoCamino(red.camino, r.porMano?.[0] ?? r.gesto), profundidad: arbol.profundidad } : null,
  });
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
    tamanoVideo = await iniciarManos(video, alDetectar);
    actualizarModo();
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
  tamanoVideo = { ancho: 640, alto: 480 };
  actualizarModo();
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

// --- modo holograma; con "Cámara de fondo", el video es el fondo de toda la pantalla si está activa (o simulada)
let tamanoVideo = { ancho: 640, alto: 480 };
const RESOLUCION_HOLOGRAMA = 2; // el lienzo de las manos se agranda para que el trazo no se vea borroso

function actualizarModo() {
  const estado = document.getElementById('camara').dataset.estado;
  const ar = modoHolograma && camaraDeFondo && (estado === 'activa' || estado === 'simulacion');
  document.body.classList.toggle('holograma', modoHolograma);
  document.body.classList.toggle('holograma-ar', ar);
  visor.fijarEstilo(modoHolograma ? 'holograma' : 'solido');
  visor.mostrarGrilla(!ar);
  etiquetas.fijarActivas(modoHolograma && conEtiquetas);
  const f = ar ? RESOLUCION_HOLOGRAMA : 1;
  if (lienzo.width !== tamanoVideo.ancho * f || lienzo.height !== tamanoVideo.alto * f) {
    lienzo.width = tamanoVideo.ancho * f;
    lienzo.height = tamanoVideo.alto * f;
  }
  for (const [clave, activo] of [['holograma', modoHolograma], ['etiquetas', modoHolograma && conEtiquetas],
    ['fondo', modoHolograma && camaraDeFondo]]) {
    document.querySelector(`[data-barra="${clave}"]`).setAttribute('aria-pressed', String(activo));
  }
  document.querySelector('[data-barra="etiquetas"]').disabled = !modoHolograma;
  document.querySelector('[data-barra="fondo"]').disabled = !modoHolograma;
}
new MutationObserver(actualizarModo).observe(document.getElementById('camara'), { attributeFilter: ['data-estado'] });

const ACCIONES_BARRA = {
  holograma: () => {
    modoHolograma = !modoHolograma;
    recordar('holograma', modoHolograma ? 'si' : 'no');
    actualizarModo();
  },
  fondo: () => {
    camaraDeFondo = !camaraDeFondo;
    recordar('camara-fondo', camaraDeFondo ? 'si' : 'no');
    actualizarModo();
  },
  etiquetas: () => {
    conEtiquetas = !conEtiquetas;
    recordar('etiquetas', conEtiquetas ? 'si' : 'no');
    actualizarModo();
  },
  despiece: () => {
    visor.guardarEstado();
    alternarDespiece();
  },
  paneles: () => {
    const ver = document.body.classList.toggle('ver-paneles');
    document.querySelector('[data-barra="paneles"]').setAttribute('aria-pressed', String(ver));
  },
  actividad: () => alternarActividad(),
};
document.getElementById('barra').addEventListener('click', (e) => {
  const accion = e.target.closest('button')?.dataset.barra;
  if (accion) ACCIONES_BARRA[accion]();
});
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'h' || e.key === 'H') ACCIONES_BARRA.holograma();
  if ((e.key === 'c' || e.key === 'C') && modoHolograma) ACCIONES_BARRA.fondo();
  if (e.key === 'l' || e.key === 'L') ACCIONES_BARRA.actividad();
});

actualizarModo();
encenderCamara();
