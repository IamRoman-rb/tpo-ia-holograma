// Estilo holograma: nube de puntos sobre la superficie, etiquetas a los costados y del dedo en la cámara a la pantalla.
// Uso (desde la raíz):  node frontend/tests/holograma.test.mjs
import {
  areaSuperficie, camaraAPantalla, colorPieza, crearAzar, distribuirEtiquetas, PALETA, puntosEnSuperficie,
} from '../js/holograma.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };
const cerca = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

// --- 1. Muestreo de la superficie
// Dos triángulos: uno de área 1 (en z = 0) y otro de área 3 (en z = 5), con índices
const pos = new Float32Array([0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 5, 3, 0, 5, 0, 2, 5]);
const indices = new Uint16Array([0, 1, 2, 3, 4, 5]);
ok(cerca(areaSuperficie(pos, indices), 4), 'área de la malla = suma de las áreas de sus triángulos');
ok(cerca(areaSuperficie(pos, null), 4), 'también sin índices');

const n = 4000;
const pts = puntosEnSuperficie(pos, indices, n, crearAzar(7));
let enGrande = 0;
let adentro = true;
for (let i = 0; i < n; i++) {
  const [x, y, z] = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
  const [ancho, alto] = z === 5 ? [3, 2] : [2, 1];
  if (z === 5) enGrande++;
  else if (z !== 0) adentro = false;
  if (x < -1e-6 || y < -1e-6 || x / ancho + y / alto > 1 + 1e-6) adentro = false;
}
ok(pts.length === n * 3, `devuelve ${n} puntos`);
ok(adentro, 'todos los puntos caen dentro de algún triángulo');
ok(Math.abs(enGrande / n - 0.75) < 0.03, `se reparten según el área (${(100 * enGrande / n).toFixed(1)} % en el triángulo de área 3/4)`);

const otra = puntosEnSuperficie(pos, indices, 50, crearAzar(7));
ok(otra.every((v, i) => v === puntosEnSuperficie(pos, indices, 50, crearAzar(7))[i]),
  'con la misma semilla sale la misma nube (no "tiembla" al recargar)');
ok(puntosEnSuperficie(new Float32Array(0), null, 10).length === 0, 'una malla vacía no da puntos');

// --- 2. Colores
ok(new Set(PALETA).size === PALETA.length, 'la paleta no repite colores');
ok(colorPieza(PALETA.length) === colorPieza(0), 'con más piezas que colores, la paleta vuelve a empezar');

// --- 3. Etiquetas a los costados
const opciones = { ancho: 1000, arriba: 80, abajo: 600, separacion: 48 };
const puntos = [
  { id: 'a', x: 300, y: 300 }, { id: 'b', x: 320, y: 310 }, { id: 'c', x: 340, y: 305 },
  { id: 'd', x: 700, y: 50 }, { id: 'e', x: 650, y: 900 },
];
const lugares = distribuirEtiquetas(puntos, opciones);
const lado = (id) => lugares.find((l) => l.id === id).lado;
ok(['a', 'b', 'c'].every((id) => lado(id) === 'izquierda') && ['d', 'e'].every((id) => lado(id) === 'derecha'),
  'cada etiqueta va del lado de su pieza');
const sinPisarse = ['izquierda', 'derecha'].every((l) => {
  const ys = lugares.filter((p) => p.lado === l).map((p) => p.y).sort((a, b) => a - b);
  return ys.every((y, i) => i === 0 || y - ys[i - 1] >= opciones.separacion - 1e-9);
});
ok(sinPisarse, 'las etiquetas de una misma columna no se pisan');
ok(lugares.every((l) => l.y >= opciones.arriba && l.y <= opciones.abajo), 'ninguna se sale del alto disponible');
const izq = lugares.filter((l) => l.lado === 'izquierda');
ok(izq.map((l) => l.id).join('') === 'acb', 'dentro de la columna respetan el orden vertical de las piezas');

const muchas = Array.from({ length: 20 }, (_, i) => ({ id: `p${i}`, x: 100, y: 300 }));
const apretadas = distribuirEtiquetas(muchas, opciones);
ok(apretadas.every((l) => l.y >= opciones.arriba - 1e-9 && l.y <= opciones.abajo + 1e-9),
  'si no entran, se reparten en el alto disponible sin salirse');

// --- 4. Del punto de la cámara a la pantalla (video a pantalla completa, espejado y recortado)
const pantalla = { anchoVideo: 640, altoVideo: 480, anchoPantalla: 1600, altoPantalla: 900 };
const centro = camaraAPantalla({ x: 0.5, y: 0.5 }, pantalla);
ok(cerca(centro.u, 0.5) && cerca(centro.v, 0.5), 'el centro de la cámara es el centro de la pantalla');
ok(camaraAPantalla({ x: 0.25, y: 0.5 }, pantalla).u > 0.5, 'espejado: lo que está a la izquierda de la imagen se ve a la derecha');
// 1600 x 900 recorta el video arriba y abajo: el borde de arriba de la imagen queda afuera de la pantalla
ok(camaraAPantalla({ x: 0.5, y: 0.05 }, pantalla).v === 0, 'lo que el recorte deja afuera se lleva al borde');
const esquina = camaraAPantalla({ x: 1, y: 0.5 }, pantalla);
ok(cerca(esquina.u, 0), 'a lo ancho no hay recorte: el borde de la imagen es el borde de la pantalla');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
