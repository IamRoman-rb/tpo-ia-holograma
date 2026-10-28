// Pruebas de la lógica de gestos con manos sintéticas (no necesita cámara).
// Uso (desde la raíz):  node frontend/tests/gestos.test.mjs
import { readFileSync } from 'node:fs';
import { normalizar } from '../js/normalize.js';
import { clasificarFotograma, crearClasificador } from '../js/gestures.js';
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

// Visor falso: registra las llamadas. piezaBajo es la pieza que "hay" bajo el cursor.
function visorFalso() {
  const llamadas = [];
  const v = {
    llamadas, piezaBajo: null, elegida: null, enAnalisis: null,
    guardarEstado: () => llamadas.push('guardar'), reiniciar: () => llamadas.push('reiniciar'),
    desplazar: () => llamadas.push('desplazar'), rotar: () => {}, escala: () => 1, escalarA: () => {},
    apuntar: (u) => (u == null ? null : v.piezaBajo), seleccionada: () => v.elegida,
    seleccionar: (id) => { v.elegida = id; llamadas.push(`seleccionar:${id}`); },
    infoPieza: (id) => ({ id }), moverPieza: (id) => llamadas.push(`mover:${id}`),
    tienePiezas: () => true, aislada: () => v.enAnalisis,
    aislar: (id) => { v.enAnalisis = id; llamadas.push(`aislar:${id}`); return { id }; },
    salirDeAislamiento: () => { const habia = !!v.enAnalisis; v.enAnalisis = null; return habia; },
    deshacer: () => { llamadas.push('deshacer'); return true; },
  };
  return v;
}

// 8. Los controles guardan el estado una sola vez al empezar cada gesto
{
  const visor = visorFalso();
  const controles = crearControles(visor);
  const mano = aImagen(POSES.pellizco, 0.2, 0.5, 0.8);
  for (let f = 0; f < 5; f++) controles('pellizco', [mano]);
  controles('ninguno', []);
  controles('puño', [mano]);
  controles('puño', [mano]);
  const sinMover = visor.llamadas.filter((l) => l !== 'desplazar');
  ok(JSON.stringify(sinMover) === JSON.stringify(['guardar', 'guardar', 'reiniciar']),
    `guarda al empezar el pellizco y antes del reinicio por puño (${sinMover.join(', ')})`);
}

// 9. Señalar una pieza 0,8 s la elige; pellizcarla la saca sola; victoria la aísla; puño sale del análisis
{
  const visor = visorFalso();
  const eventos = [];
  const controles = crearControles(visor, { alEvento: (tipo) => eventos.push(tipo) });
  const senalar = aImagen(POSES["señalar"], 0.15, 0.5, 0.5);
  const pellizco = aImagen(POSES.pellizco, 0.15, 0.5, 0.5);
  visor.piezaBajo = 'volante';
  controles('señalar', [senalar], 0);
  controles('señalar', [senalar], 500);
  ok(visor.elegida === null, 'apuntar medio segundo todavía no elige');
  controles('señalar', [senalar], 900);
  ok(visor.elegida === 'volante' && eventos.includes('seleccion'), 'apuntar 0,8 s elige la pieza');

  controles('pellizco', [pellizco], 1000);
  controles('pellizco', [aImagen(POSES.pellizco, 0.15, 0.3, 0.5)], 1050);
  ok(visor.llamadas.includes('mover:volante') && !visor.llamadas.includes('desplazar'),
    'el pellizco sobre la pieza elegida mueve solo esa pieza');

  visor.piezaBajo = null;
  controles('ninguno', [], 1100);
  controles('pellizco', [pellizco], 2500); // ya pasó más de 1 s desde que se apuntó la pieza
  controles('pellizco', [aImagen(POSES.pellizco, 0.15, 0.3, 0.5)], 2550);
  ok(visor.llamadas.includes('desplazar'), 'un pellizco lejos de la pieza mueve el objeto entero');

  // Del señalar al pellizco hay cuadros sin gesto: la pieza apuntada se recuerda un momento
  {
    const v2 = visorFalso();
    const c2 = crearControles(v2);
    v2.piezaBajo = 'vastago';
    for (const t of [0, 400, 850]) c2('señalar', [senalar], t);
    v2.piezaBajo = null;
    c2('ninguno', [], 900);
    c2('pellizco', [pellizco], 1100);
    c2('pellizco', [aImagen(POSES.pellizco, 0.15, 0.3, 0.5)], 1150);
    ok(v2.llamadas.includes('mover:vastago'), 'pasar de señalar a pellizcar (con cuadros sin gesto) toma la pieza');
  }

  controles('victoria', [senalar], 2600);
  ok(visor.enAnalisis === 'volante' && eventos.includes('aislar'), 'victoria con una pieza elegida la aísla');
  controles('puño', [senalar], 2700);
  ok(visor.enAnalisis === null && eventos.includes('salir') && !visor.llamadas.includes('reiniciar'),
    'el puño sale del análisis sin reiniciar la vista');

  visor.elegida = null;
  controles('victoria', [senalar], 2800);
  ok(eventos.at(-1) === 'despiece', 'victoria sin pieza elegida pide el despiece');
  controles('pulgar_arriba', [senalar], 2900);
  controles('pulgar_abajo', [senalar], 3000);
  ok(eventos.slice(-2).join() === 'confirmar,deshacer' && visor.llamadas.at(-1) === 'deshacer',
    'pulgar arriba confirma y pulgar abajo deshace');
}

// 10. Gestos nuevos por reglas: pulgar arriba, pulgar abajo y victoria
{
  const r = (nombre) => clasificarFotograma(aImagen(POSES[nombre], 0.15, 0.5, 0.5), 1);
  for (const nombre of ['pulgar_arriba', 'pulgar_abajo', 'victoria']) {
    if (POSES[nombre]) ok(r(nombre) === nombre, `${nombre}: ${r(nombre)}`);
  }
}

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
