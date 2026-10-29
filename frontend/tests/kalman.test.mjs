// Filtro de Kalman de los puntos de la mano: elige sus parámetros con el simulador y verifica que reduce el temblor.
// Una mano sintética se mueve (quieta, despacio y rápido) y se le suma ruido gaussiano del tamaño del de
// MediaPipe (σ = 0,002 de la imagen). Se compara contra la posición real.
import { PARAMETROS, crearFiltroManos } from '../js/kalman.js';
import { POSES, aImagen } from '../js/manos_sinteticas.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

// Generador de números pseudoaleatorios con semilla y ruido gaussiano (Box-Muller): el test es reproducible
let semilla = 7;
const azar = () => ((semilla = (semilla * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(azar() + 1e-12)) * Math.cos(2 * Math.PI * azar());
const SIGMA = 0.002;
const FPS = 30;

// Recorrido: 2 s quieta, 3 s yendo y viniendo despacio, 2 s rápido (un barrido de media pantalla por segundo).
// El primer medio segundo no se mide (el filtro arranca).
function posicion(t) {
  if (t < 2) return [0.5, 0.6];
  if (t < 5) return [0.5 + 0.1 * Math.sin((t - 2) * Math.PI * 0.6), 0.6];
  return [0.5 + 0.25 * Math.sin((t - 5) * Math.PI * 2), 0.6 + 0.05 * Math.cos((t - 5) * Math.PI * 2)];
}

function correr(parametros) {
  semilla = 7;
  const filtro = crearFiltroManos(parametros);
  const e = { quieta: [0, 0, 0], movimiento: [0, 0, 0] };   // error² medido, error² filtrado, n
  for (let k = 0; k < 7 * FPS; k++) {
    const t = k / FPS;
    const [x, y] = posicion(t);
    const real = aImagen(POSES.palma_abierta, 0.3, x, y);
    const medida = real.map((p) => ({ x: p.x + SIGMA * gauss(), y: p.y + SIGMA * gauss(), z: p.z + SIGMA * gauss() }));
    const [suave] = filtro.filtrar([medida], ['derecha'], t * 1000);
    if (t < 0.5) continue;
    const tramo = t < 2 ? e.quieta : e.movimiento;
    for (let i = 0; i < 21; i++) {
      tramo[0] += (medida[i].x - real[i].x) ** 2 + (medida[i].y - real[i].y) ** 2;
      tramo[1] += (suave[i].x - real[i].x) ** 2 + (suave[i].y - real[i].y) ** 2;
      tramo[2] += 1;
    }
  }
  const rmse = (tr, j) => Math.sqrt(tr[j] / tr[2]);
  return { quietaMedido: rmse(e.quieta, 0), quietaFiltrado: rmse(e.quieta, 1),
    movMedido: rmse(e.movimiento, 0), movFiltrado: rmse(e.movimiento, 1) };
}

// Búsqueda en grilla de q y de la adaptación: el menor temblor quieta, sin atrasarse más de 1,75 veces el ruido
const fmt = (v) => `${(v * 1000).toFixed(2)} ‰`;
const grilla = [];
for (const q of [0.005, 0.01, 0.05, 0.2, 1]) {
  for (const adaptacion of [0, 2, 10, 50, 200]) grilla.push({ q, adaptacion, ...correr({ ...PARAMETROS, q, adaptacion }) });
}
const validos = grilla.filter((g) => g.movFiltrado <= 1.75 * g.movMedido);
const mejor = validos.reduce((a, b) => (b.quietaFiltrado < a.quietaFiltrado ? b : a));
console.log(`      grilla de ${grilla.length}: mejor q = ${mejor.q}, adaptación = ${mejor.adaptacion} `
  + `(quieta ${fmt(mejor.quietaFiltrado)}, en movimiento ${fmt(mejor.movFiltrado)})`);
ok(mejor.q === PARAMETROS.q && mejor.adaptacion === PARAMETROS.adaptacion, 'el filtro usa el mejor par de la grilla');

const r = correr(PARAMETROS);
const fijo = correr({ ...PARAMETROS, adaptacion: 0 });
ok(r.quietaFiltrado < 0.6 * r.quietaMedido,
  `mano quieta: el temblor baja de ${fmt(r.quietaMedido)} a ${fmt(r.quietaFiltrado)} de la imagen`);
ok(r.movFiltrado <= 1.75 * r.movMedido, `en movimiento el atraso queda acotado: ${fmt(r.movMedido)} → ${fmt(r.movFiltrado)}`);
ok(r.movFiltrado < fijo.movFiltrado / 5,
  `la adaptación sigue los movimientos rápidos (sin adaptar, con el mismo q: ${fmt(fijo.movFiltrado)})`);

// Si la mano salta lejos (otra mano entró al cuadro), el filtro arranca de nuevo: no la arrastra
const f = crearFiltroManos();
const aca = aImagen(POSES.puño, 0.3, 0.2, 0.6);
const alla = aImagen(POSES.puño, 0.3, 0.8, 0.6);
f.filtrar([aca], ['derecha'], 0);
f.filtrar([aca], ['derecha'], 33);
const [saltada] = f.filtrar([alla], ['derecha'], 66);
ok(Math.abs(saltada[0].x - alla[0].x) < 1e-9, 'un salto grande reinicia el filtro de esa mano');

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
