// Modo simulación: genera manos sintéticas en movimiento y las pasa por el mismo camino
// que la cámara (clasificador → controles → visor). Sirve para probar y mostrar el sistema
// en equipos sin cámara. Se indica en pantalla que es una simulación.
import { POSES, aImagen } from './manos_sinteticas.js';

const TAMANO = 0.3;  // alto de la mano en fracción del cuadro
const RUIDO = 0.0015; // temblor parecido al de los landmarks reales
const FPS = 30;

// Guion que se repite: cada tramo dura `seg` segundos y devuelve las manos para t ∈ [0, 1]
const GUION = [
  { seg: 1.5, manos: () => [] },
  { seg: 4, manos: (t) => [mano('palma_abierta', 0.5 + 0.18 * Math.sin(t * 2 * Math.PI), 0.85)] },
  { seg: 3.5, manos: (t) => [mano('pellizco', 0.5 + 0.12 * Math.cos(t * 2 * Math.PI), 0.82 + 0.06 * Math.sin(t * 2 * Math.PI))] },
  { seg: 1, manos: () => [] },
  {
    seg: 3.5,
    manos: (t) => {
      const abierto = 0.2 + 0.12 * Math.sin(t * Math.PI); // se separan y se vuelven a juntar
      // espejadas para que los pellizcos queden enfrentados
      return [mano('pellizco', 0.5 - abierto, 0.85, true), mano('pellizco', 0.5 + abierto, 0.85, false)];
    },
  },
  { seg: 1, manos: () => [] },
  { seg: 1.5, manos: () => [mano('puño', 0.5, 0.8)] },
];
const DURACION = GUION.reduce((s, g) => s + g.seg, 0);

function mano(pose, x, y, espejo = false) {
  return { puntos: aImagen(POSES[pose], TAMANO, x, y, { espejo, ruido: RUIDO }), x };
}

/** Manos (y su lado) en el segundo `seg` del guion. */
export function manosEn(seg) {
  let t = seg % DURACION;
  const tramo = GUION.find((g) => (t -= g.seg) < 0) ?? GUION[0];
  const manos = tramo.manos((t + tramo.seg) / tramo.seg);
  // en la imagen sin espejar, la mano derecha del usuario aparece a la izquierda
  return { manos: manos.map((m) => m.puntos), lados: manos.map((m) => (m.x < 0.5 ? 'derecha' : 'izquierda')) };
}

/** Arranca la simulación. Devuelve una función para detenerla. */
export function iniciarSimulacion(onResultados) {
  let activo = true;
  let ultimo = 0;
  const inicio = performance.now();

  const bucle = (ahora) => {
    if (!activo) return;
    if (ahora - ultimo >= 1000 / FPS) {
      ultimo = ahora;
      const { manos, lados } = manosEn((ahora - inicio) / 1000);
      onResultados(manos, { aspecto: 640 / 480, lados, msProcesamiento: 0, simulacion: true });
    }
    requestAnimationFrame(bucle);
  };
  requestAnimationFrame(bucle);
  return () => { activo = false; };
}

/**
 * Una mano con la pose pedida, moviéndose y variando distancia, ángulo y lado (para la
 * herramienta de captura). Imita las variaciones que pide la sección 12.2.
 */
export function manoConVariaciones(pose, seg) {
  const espejo = Math.floor(seg / 6) % 2 === 1; // alterna de mano cada 6 s
  const x = 0.5 + 0.15 * Math.sin(seg * 0.9);
  const y = 0.8 + 0.05 * Math.sin(seg * 1.3);
  const tamano = 0.24 + 0.08 * Math.sin(seg * 0.5); // más cerca y más lejos de la cámara
  const rot = 0.35 * Math.sin(seg * 0.7);
  return {
    manos: [aImagen(POSES[pose], tamano, x, y, { espejo, rot, ruido: RUIDO })],
    lados: [espejo ? 'izquierda' : 'derecha'],
  };
}

/** Simulación para la captura: la pose la decide `poseActual()` en cada cuadro. */
export function iniciarSimulacionPose(poseActual, onResultados) {
  let activo = true;
  let ultimo = 0;
  const inicio = performance.now();
  const bucle = (ahora) => {
    if (!activo) return;
    if (ahora - ultimo >= 1000 / FPS) {
      ultimo = ahora;
      const { manos, lados } = manoConVariaciones(poseActual(), (ahora - inicio) / 1000);
      onResultados(manos, { aspecto: 640 / 480, lados, msProcesamiento: 0, simulacion: true });
    }
    requestAnimationFrame(bucle);
  };
  requestAnimationFrame(bucle);
  return () => { activo = false; };
}

