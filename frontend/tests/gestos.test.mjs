// Pruebas de la lógica de gestos con manos sintéticas (no necesita cámara).
// Uso (desde la raíz):  node frontend/tests/gestos.test.mjs
import { readFileSync } from 'node:fs';
import { normalizar } from '../js/normalize.js';
import { crearClasificador } from '../js/gestures.js';
import { buscarLocal } from '../js/catalogo.js';
import { ASPECTO, POSES, aImagen } from '../js/manos_sinteticas.js';
import { manosEn } from '../js/simulador.js';
import { crearHistorial } from '../js/historial.js';
import { crearControles } from '../js/controles.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

// 1. Normalización: misma mano cerca/lejos y en distinto lugar del cuadro -> mismo vector
const cerca = normalizar(aImagen(POSES.palma_abierta, 0.18, 0.4, 0.8), ASPECTO);
const lejos = normalizar(aImagen(POSES.palma_abierta, 0.07, 0.7, 0.5), ASPECTO);
const maxDif = Math.max(...cerca.flatMap((p, i) => [p.x - lejos[i].x, p.y - lejos[i].y, p.z - lejos[i].z].map(Math.abs)));
ok(maxDif < 1e-9, `normalización invariante a escala y posición (dif máx ${maxDif.toExponential(1)})`);

// 2. Reglas: cada pose a dos distancias
for (const [esperado, pose] of Object.entries(POSES)) {
  for (const [s, ox, oy] of [[0.18, 0.4, 0.8], [0.07, 0.7, 0.5]]) {
    const c = crearClasificador();
    let r;
    for (let f = 0; f < 6; f++) r = c([aImagen(pose, s, ox, oy)], ASPECTO);
    ok(r.gesto === esperado && r.confianza === 1, `${esperado} a escala ${s} -> ${r.gesto} (${r.confianza})`);
  }
}

// 3. Dos manos pellizcando -> escala_dos_manos
{
  const c = crearClasificador(); let r;
  for (let f = 0; f < 6; f++) r = c([aImagen(POSES.pellizco, .1, .3, .7), aImagen(POSES.pellizco, .1, .7, .7)], ASPECTO);
  ok(r.gesto === 'escala_dos_manos', `dos manos pellizcando -> ${r.gesto}`);
}

// 4. Un fotograma suelto no supera el umbral de 0.8
{
  const c = crearClasificador(); let r;
  for (let f = 0; f < 5; f++) r = c([aImagen(POSES.palma_abierta, .1, .5, .7)], ASPECTO);
  r = c([aImagen(POSES['puño'], .1, .5, .7)], ASPECTO);
  ok(r.gesto === 'palma_abierta' && r.confianza < 0.9, `fotograma aislado filtrado -> ${r.gesto} (${r.confianza.toFixed(2)})`);
}

// 5. Búsqueda local en el catálogo
const items = JSON.parse(readFileSync(new URL('../models/catalogo.json', import.meta.url), 'utf8'));
for (const [frase, id] of [['mostrame una válvula de acero', 'valvula_compuerta'], ['quiero ver un rulemán', 'rodamiento_bolas'],
  ['Motor', 'motor_electrico'], ['dame unos tornillos', 'tornillo_hexagonal'], ['una rueda dentada', 'engranaje_recto'], ['qué hora es', null]]) {
  const r = buscarLocal(frase, items);
  ok((r?.item.id ?? null) === id, `"${frase}" -> ${r?.item.id ?? 'sin resultado'}`);
}
// 6. El guion de la simulación produce los gestos que dice mostrar (con ruido)
for (const [seg, esperado] of [[0.75, 'ninguno'], [3.5, 'palma_abierta'], [7.25, 'pellizco'],
  [11.75, 'escala_dos_manos'], [15.25, 'puño']]) {
  const c = crearClasificador(); let r;
  for (let f = 0; f < 6; f++) r = c(manosEn(seg + f / 30).manos, ASPECTO);
  ok(r.gesto === esperado, `simulación en ${seg} s -> ${r.gesto}`);
}

// 7. Historial para "Deshacer"
{
  const h = crearHistorial(3);
  const e = (x) => ({ pos: [x, 0, 0], rotX: 0, rotY: 0, escala: 1 });
  h.guardar(e(0)); h.guardar(e(0)); h.guardar(e(1));
  ok(h.cantidad === 2, `no guarda dos veces el mismo estado (${h.cantidad})`);
  ok(h.deshacer(e(2)).pos[0] === 1 && h.deshacer(e(1)).pos[0] === 0, 'deshace en orden inverso');
  ok(h.deshacer(e(0)) === null, 'sin estados: nada para deshacer');
  h.guardar(e(5));
  ok(h.deshacer(e(5)) === null, 'un estado igual al actual no cuenta como cambio');
  for (let i = 0; i < 10; i++) h.guardar(e(i));
  ok(h.cantidad === 3, `respeta el máximo (${h.cantidad})`);
}

// 8. Los controles guardan el estado una sola vez al empezar cada gesto
{
  const llamadas = [];
  const visor = {
    guardarEstado: () => llamadas.push('guardar'), reiniciar: () => llamadas.push('reiniciar'),
    desplazar: () => {}, rotar: () => {}, escala: () => 1, escalarA: () => {},
  };
  const controles = crearControles(visor);
  const mano = aImagen(POSES.pellizco, 0.2, 0.5, 0.8);
  for (let f = 0; f < 5; f++) controles('pellizco', [mano]);
  controles('ninguno', []);
  controles('puño', [mano]);
  controles('puño', [mano]);
  ok(JSON.stringify(llamadas) === JSON.stringify(['guardar', 'guardar', 'reiniciar']),
    `guarda al empezar el pellizco y antes del reinicio por puño (${llamadas.join(', ')})`);
}

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
