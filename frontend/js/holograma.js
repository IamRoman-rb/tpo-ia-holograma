// Estilo holograma: cada pieza en un color neón, dibujada con sus aristas, una nube de puntos sobre la
// superficie y un relleno tenue. Acá están las partes que no dependen de Three.js (se prueban con node):
// la paleta, el muestreo de puntos y la distribución de las etiquetas a los costados de la pantalla.

// Colores de las piezas, en orden de desmontaje. Se eligieron para que se distingan sobre el video de la cámara.
export const PALETA = ['#2bd4ff', '#ff4fd8', '#ffd23f', '#ff5a4f', '#3dffb0', '#a77bff', '#ff9a3c', '#f2f6ff',
  '#4f8bff', '#c8ff3c'];

export const colorPieza = (i) => PALETA[i % PALETA.length];

// Generador pseudoaleatorio con semilla: la nube de puntos sale igual cada vez que se carga el modelo.
export function crearAzar(semilla = 1) {
  let s = semilla >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/** Área total de una malla de triángulos (mismos argumentos que puntosEnSuperficie). */
export function areaSuperficie(posiciones, indices) {
  const { total } = areasAcumuladas(posiciones, indices);
  return total;
}

function areasAcumuladas(posiciones, indices) {
  const cantidad = indices ? indices.length / 3 : posiciones.length / 9;
  const vertice = (t, k) => (indices ? indices[t * 3 + k] : t * 3 + k);
  const acumulada = new Float64Array(cantidad);
  let total = 0;
  for (let t = 0; t < cantidad; t++) {
    const [a, b, c] = [0, 1, 2].map((k) => vertice(t, k) * 3);
    const u = [0, 1, 2].map((d) => posiciones[b + d] - posiciones[a + d]);
    const v = [0, 1, 2].map((d) => posiciones[c + d] - posiciones[a + d]);
    total += Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]) / 2;
    acumulada[t] = total;
  }
  return { cantidad, vertice, acumulada, total };
}

/**
 * n puntos al azar sobre la superficie de una malla de triángulos, repartidos según el área (un triángulo grande
 * recibe más puntos). posiciones: [x0, y0, z0, x1, …] · indices: lista de vértices por triángulo, o null si la
 * malla no está indexada. Devuelve un Float32Array de 3·n.
 */
export function puntosEnSuperficie(posiciones, indices, n, azar = crearAzar()) {
  const { cantidad, vertice, acumulada, total } = areasAcumuladas(posiciones, indices);
  const salida = new Float32Array(n * 3);
  if (!cantidad || total === 0) return salida.subarray(0, 0);
  for (let i = 0; i < n; i++) {
    // Búsqueda binaria del triángulo según el área acumulada
    const objetivo = azar() * total;
    let lo = 0, hi = cantidad - 1;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (acumulada[m] < objetivo) lo = m + 1;
      else hi = m;
    }
    // Punto uniforme dentro del triángulo (coordenadas baricéntricas)
    let r1 = azar(), r2 = azar();
    if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
    const [a, b, c] = [0, 1, 2].map((k) => vertice(lo, k) * 3);
    for (let d = 0; d < 3; d++) {
      salida[i * 3 + d] = posiciones[a + d] + r1 * (posiciones[b + d] - posiciones[a + d])
        + r2 * (posiciones[c + d] - posiciones[a + d]);
    }
  }
  return salida;
}

/**
 * Ubica las etiquetas de las piezas en dos columnas, a la izquierda y a la derecha de la pantalla, sin que se
 * pisen. Cada pieza va a la columna de su lado y, dentro de la columna, lo más cerca posible de su altura.
 * puntos: [{ id, x, y }] en píxeles · Devuelve [{ id, lado: 'izquierda' | 'derecha', y }].
 */
export function distribuirEtiquetas(puntos, { ancho, arriba, abajo, separacion }) {
  const resultado = [];
  for (const lado of ['izquierda', 'derecha']) {
    const columna = puntos
      .filter((p) => (lado === 'izquierda' ? p.x < ancho / 2 : p.x >= ancho / 2))
      .sort((a, b) => a.y - b.y)
      .map((p) => ({ id: p.id, lado, y: Math.min(abajo, Math.max(arriba, p.y)) }));
    // Si no entran todas, se reparten parejo en el alto disponible
    const disponible = abajo - arriba;
    const sep = columna.length > 1 ? Math.min(separacion, disponible / (columna.length - 1)) : separacion;
    // De arriba hacia abajo: cada una por lo menos "sep" debajo de la anterior…
    for (let i = 1; i < columna.length; i++) columna[i].y = Math.max(columna[i].y, columna[i - 1].y + sep);
    // …y de abajo hacia arriba, para que la última no se salga
    if (columna.length) columna[columna.length - 1].y = Math.min(columna[columna.length - 1].y, abajo);
    for (let i = columna.length - 2; i >= 0; i--) columna[i].y = Math.min(columna[i].y, columna[i + 1].y - sep);
    resultado.push(...columna);
  }
  return resultado;
}

/**
 * Del punto de la cámara (coordenadas de MediaPipe, 0-1) al punto de la pantalla donde se ve, cuando el video
 * ocupa toda la pantalla espejado y recortado para cubrirla (object-fit: cover). Devuelve { u, v } en 0-1.
 */
export function camaraAPantalla(p, { anchoVideo, altoVideo, anchoPantalla, altoPantalla }) {
  const s = Math.max(anchoPantalla / anchoVideo, altoPantalla / altoVideo);
  const [w, h] = [anchoVideo * s, altoVideo * s];
  const limitar = (v) => Math.min(1, Math.max(0, v));
  return {
    u: limitar(((anchoPantalla - w) / 2 + (1 - p.x) * w) / anchoPantalla),
    v: limitar(((altoPantalla - h) / 2 + p.y * h) / altoPantalla),
  };
}
