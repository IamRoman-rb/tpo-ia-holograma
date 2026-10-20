// Clasificación de gestos: reglas geométricas (línea base) + MLP (fase 3).
import { normalizar } from './normalize.js';

// Clases de la sección 12.1
export const GESTOS = ['pellizco', 'palma_abierta', 'puño', 'señalar', 'escala_dos_manos', 'ninguno'];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Umbral provisorio: sin normalizar falla según la distancia a la cámara (ver sección 13.1).
const UMBRAL_PELLIZCO = 0.05;

function porReglas(manos) {
  if (manos.length === 0) return { gesto: 'ninguno', confianza: 0 };
  const p = normalizar(manos[0]);
  // 4 = punta del pulgar, 8 = punta del índice
  if (dist(p[4], p[8]) < UMBRAL_PELLIZCO) return { gesto: 'pellizco', confianza: 1 };
  // TODO (rol 2): palma_abierta, puño, señalar, escala_dos_manos
  return { gesto: 'ninguno', confianza: 1 };
}

export function clasificarGesto(manos) {
  // TODO (fase 3): sumar la inferencia del MLP exportado (ONNX / TF.js) y comparar.
  return porReglas(manos);
}
