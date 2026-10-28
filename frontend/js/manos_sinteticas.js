// Manos sintéticas con la misma estructura de 21 puntos que MediaPipe.
// Se usan en los tests (frontend/tests) y en el modo simulación del recuadro de cámara,
// para probar el sistema sin cámara. No reemplazan las pruebas con manos reales.

export const ASPECTO = 640 / 480;

const sumar = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

// Nudillos de índice, medio, anular y meñique (unidad = muñeca → nudillo medio)
const BASES = [[-0.35, 1, 0], [-0.1, 1, 0], [0.12, 1, 0], [0.32, 0.95, 0]];

// abrir: inclinación lateral de un dedo extendido (para la V de la victoria)
function dedo(base, modo, abrir = 0) {
  if (modo === 'ext') {
    const p = sumar(base, [abrir * 0.45, 0.45, 0]), d = sumar(p, [abrir * 0.28, 0.28, 0]);
    return [base, p, d, sumar(d, [abrir * 0.22, 0.22, 0])];
  }
  if (modo === 'dob') {
    const p = sumar(base, [0, 0, -0.45]), d = sumar(p, [0, -0.28, 0]);
    return [base, p, d, sumar(d, [0, -0.1, 0.2])];
  }
  // 'semi': índice curvado hacia el pulgar (pellizco)
  const p = sumar(base, [0, 0.4, -0.15]), d = sumar(p, [-0.05, 0.15, -0.2]);
  return [base, p, d, sumar(d, [-0.05, 0, -0.15])];
}

function mano(modos, puntaPulgar, aperturas = []) {
  const pts = [[0, 0, 0], [-0.35, 0.25, 0], [-0.6, 0.5, 0], [-0.75, 0.75, 0], puntaPulgar ?? [-0.85, 0.95, 0]];
  modos.forEach((m, i) => pts.push(...dedo(BASES[i], m, aperturas[i])));
  return pts;
}

// Gira una pose media vuelta alrededor de la muñeca (en el plano de la imagen)
const darVuelta = (pts) => pts.map(([x, y, z]) => [-x, -y, z]);
const pulgarArriba = mano(['dob', 'dob', 'dob', 'dob'], [-0.85, 1.6, 0]);

const puntaIndiceSemi = dedo(BASES[0], 'semi')[3];

export const POSES = {
  palma_abierta: mano(['ext', 'ext', 'ext', 'ext']),
  'puño': mano(['dob', 'dob', 'dob', 'dob'], [-0.2, 0.7, -0.35]),
  pellizco: mano(['semi', 'ext', 'ext', 'ext'], sumar(puntaIndiceSemi, [0.03, 0, 0])),
  'señalar': mano(['ext', 'dob', 'dob', 'dob'], [-0.1, 0.6, -0.3]),
  pulgar_arriba: pulgarArriba,
  pulgar_abajo: darVuelta(pulgarArriba),
  victoria: mano(['ext', 'ext', 'dob', 'dob'], [0.05, 0.55, -0.3], [-0.35, 0.25]),
  // mano relajada, dedos a medio doblar: la clase "ninguno"
  ninguno: mano(['semi', 'semi', 'semi', 'semi']),
};

/**
 * Lleva una pose a coordenadas de imagen como las de MediaPipe (x, y en 0..1, y hacia abajo).
 * s: tamaño de la mano en fracción del alto del cuadro · (ox, oy): posición de la muñeca.
 * espejo: invierte la mano (para simular la otra mano) · ruido: temblor aleatorio.
 */
export function aImagen(pts, s, ox, oy, { aspecto = ASPECTO, espejo = false, ruido = 0, rot = 0 } = {}) {
  const r = () => (Math.random() - 0.5) * 2 * ruido;
  const [c, sn] = [Math.cos(rot), Math.sin(rot)]; // giro de la mano en el plano, alrededor de la muñeca
  return pts.map(([x0, y0, z]) => {
    const x = (espejo ? -x0 : x0) * c - y0 * sn;
    const y = (espejo ? -x0 : x0) * sn + y0 * c;
    return { x: ox + (s * x) / aspecto + r(), y: oy - s * y + r(), z: (s * z) / aspecto };
  });
}
