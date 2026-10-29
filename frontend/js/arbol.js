// Árbol de decisión de gestos en el navegador (clasificador "arbol").
//
// Lo entrena ml/arboles/entrenar_arboles.py con HaGRID, sobre rasgos que se pueden leer: qué tan estirado está
// cada dedo, el ángulo de cada articulación, distancias entre puntas. A diferencia de la red neuronal, cada
// decisión es una cadena de preguntas ("¿el índice está estirado más de 1,2?") y el panel de actividad la
// muestra entera: es la versión aprendida de las reglas escritas a mano de gestures.js.
// frontend/tests/arbol.test.mjs verifica que los rasgos y las predicciones son las mismas que en Python.
import { medir } from './gestures.js';
import { normalizar } from './normalize.js';

export const URL_ARBOL = 'ml/arbol_gestos.json';

const restar = (a, b) => [a.x - b.x, a.y - b.y, a.z - b.z];
const norma = (v) => Math.hypot(...v);
const dist = (a, b) => norma(restar(a, b));

// Ángulo (en grados) que dobla una articulación: 0 = el dedo sigue recto, 180 = doblado del todo
function curva(p, a, b, c) {
  const u = restar(p[b], p[a]);
  const v = restar(p[c], p[b]);
  const coseno = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / ((norma(u) * norma(v)) || 1);
  return (Math.acos(Math.max(-1, Math.min(1, coseno))) * 180) / Math.PI;
}

// Mismo orden y mismas cuentas que ml/arboles/rasgos_arbol.py
export const RASGOS = ['pulgarIndice', 'pulgarAfuera', 'pulgarVertical', 'separacionV', 'indice', 'medio', 'anular',
  'menique', 'curvaPulgar', 'curvaIndice', 'curvaMedio', 'curvaAnular', 'curvaMenique', 'pulgarMedio',
  'pulgarAnular', 'pulgarMenique', 'separacionMedioAnular', 'separacionAnularMenique', 'indiceVertical'];

/** Rasgos de una mano ya normalizada (normalize.js), como objeto { nombre: valor }. */
export function rasgosArbol(p) {
  return {
    ...medir(p),
    curvaPulgar: curva(p, 2, 3, 4),
    curvaIndice: curva(p, 5, 6, 8),
    curvaMedio: curva(p, 9, 10, 12),
    curvaAnular: curva(p, 13, 14, 16),
    curvaMenique: curva(p, 17, 18, 20),
    pulgarMedio: dist(p[4], p[12]),
    pulgarAnular: dist(p[4], p[16]),
    pulgarMenique: dist(p[4], p[20]),
    separacionMedioAnular: dist(p[12], p[16]),
    separacionAnularMenique: dist(p[16], p[20]),
    indiceVertical: (p[8].y - p[5].y) / (dist(p[5], p[8]) || 1),
  };
}

// Nombres para leer el camino en el panel
export const NOMBRES = {
  pulgarIndice: 'pulgar-índice', pulgarAfuera: 'pulgar afuera', pulgarVertical: 'pulgar hacia abajo',
  separacionV: 'separación índice-medio', indice: 'índice estirado', medio: 'medio estirado',
  anular: 'anular estirado', menique: 'meñique estirado', curvaPulgar: 'curva del pulgar',
  curvaIndice: 'curva del índice', curvaMedio: 'curva del medio', curvaAnular: 'curva del anular',
  curvaMenique: 'curva del meñique', pulgarMedio: 'pulgar-medio', pulgarAnular: 'pulgar-anular',
  pulgarMenique: 'pulgar-meñique', separacionMedioAnular: 'separación medio-anular',
  separacionAnularMenique: 'separación anular-meñique', indiceVertical: 'índice hacia abajo',
};

/**
 * Recorre el árbol desde la raíz. Devuelve { probabilidades, camino: [{ rasgo, umbral, valor, menor }], hoja }.
 * modelo.nodos: arreglos paralelos como los de scikit-learn (izq/der = -1 en las hojas).
 */
export function recorrer(modelo, valores) {
  const { izq, der, rasgo, umbral, valor } = modelo.nodos;
  const camino = [];
  let n = 0;
  while (izq[n] !== -1) {
    const nombre = modelo.rasgos[rasgo[n]];
    const v = valores[nombre];
    const menor = v <= umbral[n];
    camino.push({ rasgo: nombre, umbral: umbral[n], valor: v, menor });
    n = menor ? izq[n] : der[n];
  }
  return { probabilidades: valor[n], camino, hoja: n };
}

/** Clasificador de una mano: { gesto, probabilidad, probabilidades, camino }, con el umbral de confianza. */
export function crearClasificadorArbol(modelo) {
  const umbral = modelo.umbral_confianza ?? 0;
  function clasificarMano(puntos, aspecto) {
    const { probabilidades: p, camino } = recorrer(modelo, rasgosArbol(normalizar(puntos, aspecto)));
    let k = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
    return { gesto: p[k] >= umbral ? modelo.clases[k] : 'ninguno', probabilidad: p[k], probabilidades: p, camino };
  }
  clasificarMano.clases = modelo.clases;
  clasificarMano.umbral = umbral;
  clasificarMano.profundidad = modelo.profundidad;
  return clasificarMano;
}

/** "índice estirado > 1,20 → pulgar afuera ≤ 0,62 → señalar" (para el panel y para explicar por voz). */
export function textoCamino(camino, gesto) {
  const fmt = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)).replace('.', ',');
  return [...camino.map((c) => `${NOMBRES[c.rasgo] ?? c.rasgo} ${c.menor ? '≤' : '>'} ${fmt(c.umbral)}`), gesto]
    .join(' → ');
}

export async function cargarArbol(url = URL_ARBOL) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`No se pudo cargar el árbol de decisión (${r.status})`);
  return crearClasificadorArbol(await r.json());
}
