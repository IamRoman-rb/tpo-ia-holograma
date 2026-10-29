// Inferencia del MLP de gestos en el navegador (sección 5.2.b).
//
// Versión 3 (ml/evolucion/neuroevolucion.py): la arquitectura la diseña un algoritmo genético y puede tener
// conexiones residuales, normalización por capa (LayerNorm) y activaciones GELU o SiLU.
//
// El modelo se entrena en Python (ml/entrenar_mlp.py, con landmarks de HaGRID) y se exporta como JSON:
// { clases, umbral_confianza, capas: [{ W: [entrada][salida], b: [salida] }], activacion: 'relu' }.
// Son unas pocas multiplicaciones de matrices: no hace falta una librería. frontend/tests/mlp.test.mjs
// verifica que da las mismas probabilidades que scikit-learn.
import { aVector, normalizar } from './normalize.js';

export const URL_MODELO = 'ml/mlp_gestos.json';

// Los pesos pueden venir como listas (versión 1) o como float32 en base64 (versión 2: la mitad de tamaño).
// En los dos casos se devuelve una lista de filas, para que la cuenta de abajo sea la misma.
function matriz(m) {
  if (Array.isArray(m)) return m;
  const binario = typeof atob === 'function' ? atob(m.float32) : Buffer.from(m.float32, 'base64').toString('binary');
  const bytes = new Uint8Array(binario.length);
  for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
  const datos = new Float32Array(bytes.buffer);
  const [filas, cols = 0] = m.forma;
  if (!cols) return datos;
  return Array.from({ length: filas }, (_, i) => datos.subarray(i * cols, (i + 1) * cols));
}

const leerCapas = (modelo) => modelo.capas.map(({ W, b }) => ({ W: matriz(W), b: Array.from(matriz(b)) }));
const leerLineal = ({ W, b }) => ({ W: matriz(W), b: Array.from(matriz(b)) });
const leerNorma = (n) => (n ? { g: Array.from(matriz(n.g)), b: Array.from(matriz(n.b)), eps: n.eps ?? 1e-5 } : null);

// Activaciones de la versión 3 (las elige la neuroevolución). GELU con la aproximación tanh, igual que PyTorch.
const ACTIVACIONES = {
  relu: (v) => (v > 0 ? v : 0),
  gelu: (v) => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))),
  silu: (v) => v / (1 + Math.exp(-v)),
};

function lineal(a, { W, b }) {
  const salida = b.slice();
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    if (ai === 0) continue;
    const fila = W[i];
    for (let j = 0; j < salida.length; j++) salida[j] += ai * fila[j];
  }
  return salida;
}

// LayerNorm: resta la media, divide por el desvío y aplica la escala y el corrimiento aprendidos
function normalizarCapa(a, { g, b, eps }) {
  const media = a.reduce((s, v) => s + v, 0) / a.length;
  const varianza = a.reduce((s, v) => s + (v - media) ** 2, 0) / a.length;
  const d = 1 / Math.sqrt(varianza + eps);
  return a.map((v, i) => (v - media) * d * g[i] + b[i]);
}

function softmax(a) {
  const maximo = Math.max(...a);
  const e = a.map((v) => Math.exp(v - maximo));
  const suma = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / suma);
}

/** Versión 3: entrada → bloques (LayerNorm opcional, lineal, activación y conexión residual) → salida. */
function crearPropagacionV3(modelo) {
  const f = ACTIVACIONES[modelo.activacion] ?? ACTIVACIONES.relu;
  const entrada = leerLineal(modelo.entrada_capa);
  const bloques = modelo.bloques.map((bl) => ({ ...leerLineal(bl), norma: leerNorma(bl.norma), residual: bl.residual }));
  const salida = { ...leerLineal(modelo.salida), norma: leerNorma(modelo.salida.norma) };
  return function propagar(vector) {
    let a = lineal(vector, entrada).map(f);
    const activaciones = [vector, a];
    for (const bl of bloques) {
      const h = lineal(bl.norma ? normalizarCapa(a, bl.norma) : a, bl).map(f);
      a = bl.residual ? a.map((v, i) => v + h[i]) : h;
      activaciones.push(a);
    }
    const z = lineal(salida.norma ? normalizarCapa(a, salida.norma) : a, salida);
    activaciones.push(z);
    return { probabilidades: softmax(z), activaciones };
  };
}

/**
 * Devuelve propagar(vector de 63) → { probabilidades, activaciones }. activaciones tiene una lista por capa
 * (la entrada, cada capa oculta después de la ReLU y la salida antes del softmax): el panel de actividad las dibuja.
 */
export function crearPropagacion(modelo) {
  if (modelo.version === 3) return crearPropagacionV3(modelo);
  const capas = leerCapas(modelo);
  const ultima = capas.length - 1;
  return function propagar(vector) {
    let a = vector;
    const activaciones = [vector];
    capas.forEach(({ W, b }, k) => {
      const salida = b.slice();
      for (let i = 0; i < a.length; i++) {
        const ai = a[i];
        if (ai === 0) continue;
        const fila = W[i];
        for (let j = 0; j < salida.length; j++) salida[j] += ai * fila[j];
      }
      a = k < ultima ? salida.map((v) => (v > 0 ? v : 0)) : salida;
      activaciones.push(a);
    });
    const maximo = Math.max(...a);
    const e = a.map((v) => Math.exp(v - maximo));
    const suma = e.reduce((s, v) => s + v, 0);
    return { probabilidades: e.map((v) => v / suma), activaciones };
  };
}

/** Devuelve predecir(vector de 63) → probabilidades, en el orden de modelo.clases. */
export function crearPredictor(modelo) {
  const propagar = crearPropagacion(modelo);
  return (vector) => propagar(vector).probabilidades;
}

/**
 * Clasificador de una mano con el MLP: { gesto, probabilidad, probabilidades, activaciones }. Si la probabilidad
 * no llega al umbral elegido con validación, la mano cuenta como "ninguno" (así se evitan acciones no pedidas).
 */
export function crearClasificadorMlp(modelo) {
  const propagar = crearPropagacion(modelo);
  const umbral = modelo.umbral_confianza ?? 0;
  function clasificarMano(puntos, aspecto) {
    const { probabilidades: p, activaciones } = propagar(aVector(normalizar(puntos, aspecto)));
    let k = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
    return { gesto: p[k] >= umbral ? modelo.clases[k] : 'ninguno', probabilidad: p[k], probabilidades: p, activaciones };
  }
  clasificarMano.clases = modelo.clases;
  clasificarMano.umbral = umbral;
  return clasificarMano;
}

export async function cargarMlp(url = URL_MODELO) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`No se pudo cargar el MLP (${r.status})`);
  return crearClasificadorMlp(await r.json());
}

/**
 * Las capas lineales de la red en orden (para dibujarla en red.html): [{ W: filas [entrada][salida], b }].
 * Versión 3: la de entrada, la de cada bloque y la de salida; versiones 1 y 2: modelo.capas.
 */
export function capasLineales(modelo) {
  if (modelo.version === 3) return [modelo.entrada_capa, ...modelo.bloques, modelo.salida].map(leerLineal);
  return leerCapas(modelo);
}
