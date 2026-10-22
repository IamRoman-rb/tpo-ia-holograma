// Herramienta de captura del dataset de gestos (sección 12 de la propuesta).
//
// Graba los 21 landmarks de UNA mano por fotograma con su etiqueta y los descarga como CSV:
//   clase, persona, mano, aspecto, x0, y0, z0, ..., x20, y20, z20
// Las coordenadas se guardan tal como las entrega MediaPipe (sin normalizar): la
// normalización se aplica al entrenar (ml/preparar.py), igual que en el navegador.
// "aspecto" (ancho / alto del video) hace falta para normalizar.
//
// El MLP clasifica posturas de una mano. El escalado con dos manos no se graba: se
// obtiene combinando el resultado de las dos manos (ambas en pellizco).
import { iniciarManos } from './hands.js';
import { crearOverlay } from './overlay.js';
import { iniciarSimulacionPose } from './simulador.js';
import { mostrarCamara, mostrarFps } from './ui.js';
import { CLASES, PERSONA_SIMULADA, aCsv } from './dataset.js';

const INTERVALO_MS = 66;  // ~15 muestras por segundo: evita guardar fotogramas casi idénticos
const CUENTA_S = 3;       // cuenta regresiva para acomodar la mano antes de grabar

const $ = (id) => document.getElementById(id);
const video = $('webcam');
const lienzo = $('landmarks');
const overlay = crearOverlay(lienzo);

const estado = {
  persona: '',
  meta: 100,
  clase: CLASES[0].id,
  muestras: [],     // { clase, persona, mano, aspecto, puntos }
  tomas: [],        // { inicio, clase }: para deshacer la última grabación
  grabando: false,
  enCuenta: false,
  ultimaMuestra: 0,
  guardadas: 0,
  simulacion: false,
};
let detenerSimulacion = null;
let ultimoCuadro = 0;
let fps = 0;

// ------------------------------------------------------------------ paso 1: consentimiento

$('form-consentimiento').addEventListener('submit', (e) => {
  e.preventDefault();
  estado.persona = $('persona').value.trim();
  estado.meta = Number($('meta').value) || 100;
  $('persona-actual').textContent = estado.persona;
  $('paso-consentimiento').hidden = true;
  $('paso-captura').hidden = false;
  dibujarClases();
  encenderCamara();
});

// ------------------------------------------------------------------ clases y progreso

const cuenta = (clase) => estado.muestras.filter((m) => m.clase === clase).length;

function dibujarClases() {
  $('clases').replaceChildren(...CLASES.map((c, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'clase';
    b.setAttribute('role', 'radio');
    b.dataset.clase = c.id;
    b.innerHTML = `<span class="nombre">${i + 1}. ${c.nombre}</span><span class="cuenta"></span>
      <span class="progreso"><div></div></span>`;
    b.addEventListener('click', () => elegirClase(c.id));
    return b;
  }));
  elegirClase(estado.clase);
}

function elegirClase(id) {
  if (estado.grabando || estado.enCuenta) return;
  estado.clase = id;
  $('descripcion-clase').textContent = CLASES.find((c) => c.id === id).descripcion;
  actualizarProgreso();
}

function actualizarProgreso() {
  for (const b of $('clases').children) {
    const n = cuenta(b.dataset.clase);
    b.setAttribute('aria-checked', String(b.dataset.clase === estado.clase));
    b.querySelector('.cuenta').textContent = `${n} / ${estado.meta}`;
    b.querySelector('.progreso div').style.width = `${Math.min(100, (100 * n) / estado.meta)}%`;
    b.classList.toggle('completa', n >= estado.meta);
    b.disabled = estado.grabando || estado.enCuenta;
  }
  $('total-muestras').textContent = `${estado.muestras.length} muestras`;
  $('descargar').disabled = estado.muestras.length === 0;
  $('deshacer').disabled = estado.tomas.length === 0 || estado.grabando || estado.enCuenta;
}

// ------------------------------------------------------------------ grabación

function mensaje(texto, grabando = false) {
  $('estado-grabacion').textContent = texto;
  $('estado-grabacion').classList.toggle('grabando', grabando);
}

function alternarGrabacion() {
  if (estado.enCuenta) return;
  if (estado.grabando) return detener('Grabación detenida.');
  if (cuenta(estado.clase) >= estado.meta) {
    return mensaje('Ya completaste este gesto. Elegí otro o subí la meta.');
  }
  estado.enCuenta = true;
  actualizarProgreso();
  let restante = CUENTA_S;
  $('cuenta').hidden = false;
  $('cuenta').textContent = restante;
  mensaje('Acomodá la mano…');
  const intervalo = setInterval(() => {
    restante -= 1;
    if (restante > 0) {
      $('cuenta').textContent = restante;
      return;
    }
    clearInterval(intervalo);
    $('cuenta').hidden = true;
    estado.enCuenta = false;
    estado.grabando = true;
    estado.tomas.push({ inicio: estado.muestras.length, clase: estado.clase });
    $('grabar').textContent = 'Detener (espacio)';
    mensaje(`Grabando «${nombreClase()}»… mové la mano y cambiá la distancia.`, true);
    actualizarProgreso();
  }, 1000);
}

function detener(texto) {
  estado.grabando = false;
  const toma = estado.tomas.at(-1);
  if (toma && toma.inicio === estado.muestras.length) estado.tomas.pop(); // toma vacía
  $('grabar').textContent = 'Grabar (espacio)';
  mensaje(texto);
  actualizarProgreso();
}

function deshacer() {
  const toma = estado.tomas.pop();
  if (!toma) return;
  const quitadas = estado.muestras.length - toma.inicio;
  estado.muestras.length = toma.inicio;
  estado.guardadas = Math.min(estado.guardadas, estado.muestras.length);
  mensaje(`Se descartaron ${quitadas} muestras de «${CLASES.find((c) => c.id === toma.clase).nombre}».`);
  actualizarProgreso();
}

const nombreClase = () => CLASES.find((c) => c.id === estado.clase).nombre;

function alDetectar(manos, info) {
  const lados = info.lados ?? [];
  const etiquetas = manos.map((_, i) => {
    const lado = lados[i] ? `${lados[i][0].toUpperCase()}${lados[i].slice(1)}` : 'Mano';
    return estado.grabando ? `${lado} · grabando ${nombreClase().toLowerCase()}` : lado;
  });
  overlay.dibujar(manos, lados, { gesto: 'ninguno', porMano: [] }, 'ninguno', {
    simulacion: info.simulacion,
    etiquetas,
    color: estado.grabando ? '#ff6b6b' : null,
  });

  const ahora = performance.now();
  if (ultimoCuadro) fps = 0.9 * fps + 0.1 * (1000 / (ahora - ultimoCuadro));
  ultimoCuadro = ahora;
  mostrarFps(fps, info.msProcesamiento);

  const aviso = manos.length > 1 ? 'Hay dos manos en cuadro: se graba solo cuando hay una.'
    : estado.grabando && manos.length === 0 ? 'No se ve ninguna mano.' : '';
  $('aviso-mano').textContent = aviso;
  $('aviso-mano').hidden = !aviso;

  if (!estado.grabando || manos.length !== 1 || ahora - estado.ultimaMuestra < INTERVALO_MS) return;
  estado.ultimaMuestra = ahora;
  estado.muestras.push({
    clase: estado.clase,
    persona: info.simulacion ? PERSONA_SIMULADA : estado.persona,
    mano: lados[0] ?? '',
    aspecto: info.aspecto,
    puntos: manos[0],
  });
  const n = cuenta(estado.clase);
  if (n >= estado.meta) detener(`Completaste «${nombreClase()}» (${n} muestras).`);
  else {
    mensaje(`Grabando «${nombreClase()}»… ${n} / ${estado.meta}`, true);
    if (n % 10 === 0) actualizarProgreso();
  }
}

$('grabar').addEventListener('click', alternarGrabacion);
$('deshacer').addEventListener('click', deshacer);

addEventListener('keydown', (e) => {
  if ($('paso-captura').hidden || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    e.preventDefault();
    alternarGrabacion();
  } else if (/^[1-5]$/.test(e.key)) {
    elegirClase(CLASES[Number(e.key) - 1].id);
  }
});

// ------------------------------------------------------------------ descarga

$('descargar').addEventListener('click', () => {
  const simulado = estado.muestras.some((m) => m.persona === PERSONA_SIMULADA);
  const fecha = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
  const enlace = document.createElement('a');
  enlace.href = URL.createObjectURL(new Blob([aCsv(estado.muestras)], { type: 'text/csv' }));
  enlace.download = `${simulado ? 'SIMULADO_' : ''}gestos_${estado.persona}_${fecha}.csv`;
  enlace.click();
  URL.revokeObjectURL(enlace.href);
  estado.guardadas = estado.muestras.length;
  mensaje(`Descargado: ${enlace.download}. Copialo a ml/data/.`);
});

addEventListener('beforeunload', (e) => {
  if (estado.muestras.length > estado.guardadas) e.preventDefault(); // hay muestras sin descargar
});

// ------------------------------------------------------------------ cámara y simulación

async function encenderCamara() {
  detenerSimulacion?.();
  detenerSimulacion = null;
  estado.simulacion = false;
  overlay.limpiar();
  mostrarCamara('iniciando');
  try {
    const { ancho, alto } = await iniciarManos(video, alDetectar);
    lienzo.width = ancho;
    lienzo.height = alto;
    mostrarCamara('activa');
  } catch (err) {
    console.error(err);
    const motivo = err.name === 'NotAllowedError' ? 'No hay permiso para usar la cámara.'
      : err.name === 'NotFoundError' ? 'No se encontró ninguna cámara.' : 'No se pudo iniciar la cámara.';
    mostrarCamara('error', `${motivo} Sin cámara no se pueden grabar datos reales.`);
  }
}

$('camara-reintentar').addEventListener('click', encenderCamara);
$('camara-simular').addEventListener('click', () => {
  lienzo.width = 640;
  lienzo.height = 480;
  estado.simulacion = true;
  mostrarCamara('simulacion');
  mensaje('Simulación: las muestras se marcan como "simulacion" y el entrenamiento las descarta.');
  detenerSimulacion = iniciarSimulacionPose(() => estado.clase, alDetectar);
});
$('camara-detener').addEventListener('click', () => {
  if (estado.grabando) detener('Grabación detenida.');
  detenerSimulacion?.();
  detenerSimulacion = null;
  overlay.limpiar();
  mostrarCamara('error', 'Simulación detenida.');
});
