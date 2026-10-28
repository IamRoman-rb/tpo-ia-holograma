// Clasificación de gestos por reglas geométricas (línea base de la sección 5.2.a).
// En la fase 3 se suma el MLP y se comparan ambos enfoques.
import { normalizar } from './normalize.js';

// Clases de la sección 12.1
export const GESTOS = ['pellizco', 'palma_abierta', 'puño', 'señalar', 'pulgar_arriba', 'pulgar_abajo', 'victoria',
  'escala_dos_manos', 'ninguno'];

// Umbrales sobre landmarks normalizados (unidad = muñeca → nudillo medio).
// Se calibran con el modo de depuración (tecla D), que muestra las métricas en vivo.
export const UMBRALES = {
  pellizcoEntrar: 0.35,   // distancia pulgar-índice para empezar un pellizco
  pellizcoSalir: 0.5,     // histéresis: hay que abrir más para soltarlo (evita parpadeos)
  dedoExtendido: 1.2,     // punta-muñeca / articulación media-muñeca por encima de esto
  dedoDoblado: 1.0,       // ... y por debajo de esto
  // Gestos nuevos, calibrados con la partición de entrenamiento de HaGRID (ml/explorar_metricas.mjs)
  pulgarAfuera: 0.62,     // pulgar separado del puño (puño: p95 0,60 · pulgar arriba: p05 0,62)
  pulgarVertical: 0.5,    // coseno con la vertical para decir que apunta arriba o abajo
  separacionV: 0.42,      // puntas del índice y del medio separadas (victoria: p05 0,45)
  ventana: 6,             // fotogramas considerados para la confianza
};

const DEDOS = [
  { nombre: 'indice', punta: 8, medio: 6 },
  { nombre: 'medio', punta: 12, medio: 10 },
  { nombre: 'anular', punta: 16, medio: 14 },
  { nombre: 'menique', punta: 20, medio: 18 },
];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

// Métricas de una mano normalizada (unidad = muñeca → nudillo medio):
//  - indice … menique: qué tan estirado está cada dedo (> 1 estirado, < 1 doblado hacia la palma);
//  - pulgarIndice: distancia entre las puntas del pulgar y el índice (pellizco);
//  - pulgarAfuera: distancia de la punta del pulgar al nudillo del índice (pulgar separado del puño);
//  - pulgarVertical: hacia dónde apunta el pulgar (< 0 arriba, > 0 abajo; la y de la imagen crece hacia abajo);
//  - separacionV: distancia entre las puntas del índice y el medio (la V de la victoria).
export function medir(p) {
  const relaciones = {};
  for (const d of DEDOS) relaciones[d.nombre] = dist(p[d.punta], p[0]) / dist(p[d.medio], p[0]);
  return {
    pulgarIndice: dist(p[4], p[8]),
    pulgarAfuera: dist(p[4], p[5]),
    pulgarVertical: (p[4].y - p[2].y) / (dist(p[4], p[2]) || 1),
    separacionV: dist(p[8], p[12]),
    ...relaciones,
  };
}

function clasificarMano(m, pellizcaba) {
  const r = [m.indice, m.medio, m.anular, m.menique];
  const doblados = r.filter((v) => v < UMBRALES.dedoDoblado).length;
  const extendidos = r.filter((v) => v > UMBRALES.dedoExtendido).length;
  const umbralPellizco = pellizcaba ? UMBRALES.pellizcoSalir : UMBRALES.pellizcoEntrar;

  // El orden importa: en un puño el pulgar también queda cerca del índice.
  if (doblados === 4) {
    // Puño con el pulgar afuera: pulgar arriba o abajo según hacia dónde apunte
    if (m.pulgarAfuera > UMBRALES.pulgarAfuera) {
      if (m.pulgarVertical < -UMBRALES.pulgarVertical) return 'pulgar_arriba';
      if (m.pulgarVertical > UMBRALES.pulgarVertical) return 'pulgar_abajo';
    }
    return 'puño';
  }
  if (m.pulgarIndice < umbralPellizco) return 'pellizco';
  if (extendidos === 4) return 'palma_abierta';
  if (m.indice > UMBRALES.dedoExtendido && m.medio < UMBRALES.dedoDoblado
      && m.anular < UMBRALES.dedoDoblado && m.menique < UMBRALES.dedoDoblado) return 'señalar';
  if (m.indice > UMBRALES.dedoExtendido && m.medio > UMBRALES.dedoExtendido
      && m.anular < UMBRALES.dedoDoblado && m.menique < UMBRALES.dedoDoblado
      && m.separacionV > UMBRALES.separacionV) return 'victoria';
  return 'ninguno';
}

// Clasificación de un solo fotograma de una mano, sin histéresis ni ventana temporal.
// Es la que se compara contra el MLP fotograma a fotograma (ml/evaluar_reglas.mjs).
export function clasificarFotograma(puntos, aspecto) {
  return clasificarMano(medir(normalizar(puntos, aspecto)), false);
}

/**
 * Clasificador con ventana temporal. Por defecto usa las reglas; con opciones.mlp (frontend/js/mlp.js) cada mano
 * la clasifica la red neuronal y las reglas solo aportan las métricas del panel de depuración.
 */
export function crearClasificador({ mlp = null } = {}) {
  let historial = [];
  let pellizcaban = [false, false];

  return function clasificar(manos, aspecto) {
    if (manos.length === 0) {
      historial = [];
      pellizcaban = [false, false];
      return { gesto: 'ninguno', confianza: 0, metricas: null };
    }

    const metricas = manos.slice(0, 2).map((puntos) => medir(normalizar(puntos, aspecto)));
    // Con el MLP, un pellizco que ya empezó se mantiene mientras el pulgar y el índice sigan cerca
    // (la misma histéresis de las reglas): así no se suelta la pieza a mitad de un arrastre.
    // La salida completa de la red (probabilidades y activaciones) queda en "red" para el panel de actividad.
    const red = mlp ? manos.slice(0, 2).map((puntos) => mlp(puntos, aspecto)) : null;
    const porMano = mlp
      ? red.map((salida, i) => (pellizcaban[i] && metricas[i].pulgarIndice < UMBRALES.pellizcoSalir
        ? 'pellizco' : salida.gesto))
      : metricas.map((m, i) => clasificarMano(m, pellizcaban[i]));
    pellizcaban = [porMano[0] === 'pellizco', porMano[1] === 'pellizco'];

    const crudo = porMano.length === 2 && porMano.every((g) => g === 'pellizco') ? 'escala_dos_manos' : porMano[0];

    // Confianza de las reglas: proporción de los últimos fotogramas que coinciden.
    // Filtra los cambios de un solo fotograma, que son la principal fuente de falsos positivos.
    historial.push(crudo);
    if (historial.length > UMBRALES.ventana) historial.shift();
    const votos = {};
    for (const g of historial) votos[g] = (votos[g] ?? 0) + 1;
    const [gesto, n] = Object.entries(votos).sort((a, b) => b[1] - a[1])[0];

    return { gesto, confianza: n / UMBRALES.ventana, crudo, porMano, metricas, red };
  };
}
