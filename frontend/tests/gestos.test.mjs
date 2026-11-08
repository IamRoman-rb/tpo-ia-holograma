// Pruebas de la lógica de gestos con manos sintéticas (no necesita cámara).
// Uso (desde la raíz):  node frontend/tests/gestos.test.mjs
import { readFileSync } from 'node:fs';
import { normalizar } from '../js/normalize.js';
import { clasificarFotograma, crearClasificador } from '../js/gestures.js';
import { buscarLocal } from '../js/catalogo.js';
import { ASPECTO, POSES, aImagen } from '../js/manos_sinteticas.js';
import { manosEn } from '../js/simulador.js';
import { crearHistorial } from '../js/historial.js';
import { crearControles, seguir } from '../js/controles.js';
import { dentroDePantalla } from '../js/arrastrable.js';
import { AJUSTES_POR_DEFECTO, normalizarAjustes } from '../js/ajustes.js';

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
  // Al bajar una mano, los votos anteriores de "dos manos" no cuentan: con una sola mano no se puede escalar
  // (antes salía escala_dos_manos tres cuadros más, los controles buscaban la segunda mano y se colgaba la cámara)
  const gestos = [];
  for (let f = 0; f < 6; f++) gestos.push(c([aImagen(POSES.pellizco, .1, .3, .7)], ASPECTO).gesto);
  ok(!gestos.includes('escala_dos_manos'), `al soltar una mano no queda "dos manos" (${[...new Set(gestos)].join(', ')})`);
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
for (const [seg, esperado, manosEsperadas] of [[0.75, 'ninguno', 0], [3.5, 'palma_abierta', 2],
  [6.5, 'palma_abierta', 1], [9.25, 'pellizco', 1], [13.75, 'escala_dos_manos', 2], [17.25, 'puño', 1]]) {
  const c = crearClasificador(); let r;
  for (let f = 0; f < 6; f++) r = c(manosEn(seg + f / 30).manos, ASPECTO);
  const porMano = r.porMano ?? [];
  ok(r.gesto === esperado && porMano.length === manosEsperadas, `simulación en ${seg} s -> ${r.gesto} con ${porMano.length} mano(s)`);
  if (manosEsperadas === 2 && esperado === 'palma_abierta') {
    ok(porMano.every((g) => g === 'palma_abierta'), `las dos manos de la simulación tienen la palma abierta (${porMano})`);
  }
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
    desplazar: () => llamadas.push('desplazar'), giro: 0, rotar: (dx) => { v.giro += dx; llamadas.push('rotar'); }, impulso: null,
    impulsar: (vx, vy) => { v.impulso = [vx, vy]; llamadas.push('impulsar'); }, escala: () => 1, escalarA: () => {},
    arrastrarCamara: (dx, dy) => llamadas.push(['camara', dx, dy]),
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

// 11. Un gesto sin las manos que necesita no hace nada (ni rompe): ningún cuadro debe cortar el bucle de la cámara
{
  const visor = visorFalso();
  const controles = crearControles(visor);
  const mano = aImagen(POSES.pellizco, 0.12, 0.3, 0.7);
  let error = null;
  try {
    for (let f = 0; f < 3; f++) controles('escala_dos_manos', [mano, aImagen(POSES.pellizco, 0.12, 0.7, 0.7)]);
    controles('escala_dos_manos', [mano]);
    for (const g of ['pellizco', 'palma_abierta', 'señalar', 'puño']) controles(g, []);
    controles('pellizco', [mano]);
  } catch (e) { error = e; }
  ok(!error, `escalar con una mano o mover sin manos no tira error${error ? ` (${error.message})` : ''}`);
  ok(!visor.llamadas.includes('reiniciar'), 'un puño sin mano no reinicia la vista');
}

// 12. Una palma abierta desplaza la vista (como arrastrar con el mouse); dos palmas rotan el objeto
{
  const visor = visorFalso();
  const controles = crearControles(visor);
  const palma = (x, y = 0.6) => aImagen(POSES.palma_abierta, 0.2, x, y);
  // Una palma que va hacia la derecha del usuario (la x de la cámara baja) y hacia abajo
  controles('palma_abierta', [palma(0.6, 0.5)], 0, ['palma_abierta']);
  controles('palma_abierta', [palma(0.5, 0.6)], 33, ['palma_abierta']);
  const camara = visor.llamadas.filter((l) => Array.isArray(l));
  ok(camara.length === 1 && camara[0][1] > 0 && camara[0][2] > 0 && !visor.llamadas.includes('rotar'),
    `una palma arrastra la vista hacia donde va la mano, sin rotar (${JSON.stringify(camara)})`);
  ok(!visor.llamadas.includes('guardar'), 'desplazarse con la palma no es un cambio del objeto para deshacer');

  const v2 = visorFalso();
  const c2 = crearControles(v2);
  const dos = ['palma_abierta', 'palma_abierta'];
  c2('palma_abierta', [palma(0.3), palma(0.7)], 0, dos);
  c2('palma_abierta', [palma(0.25), palma(0.65)], 33, dos);
  ok(v2.llamadas.includes('rotar') && !v2.llamadas.some(Array.isArray), 'dos palmas rotan el objeto y no mueven la vista');
  ok(v2.llamadas.filter((l) => l === 'guardar').length === 1, 'rotar con dos palmas se guarda una vez para deshacer');

  // Un cuadro suelto con una sola palma no cambia de modo (histéresis)
  c2('palma_abierta', [palma(0.25), palma(0.65)], 66, ['palma_abierta', 'ninguno']);
  c2('palma_abierta', [palma(0.2), palma(0.6)], 100, dos);
  c2('palma_abierta', [palma(0.15), palma(0.55)], 133, dos);
  ok(!v2.llamadas.some(Array.isArray), 'un cuadro con la segunda mano mal clasificada no desplaza la vista');
  // Con la otra mano cerrada, pasa a desplazar
  for (let f = 0; f < 5; f++) c2('palma_abierta', [palma(0.5 - f * 0.03), palma(0.6)], 200 + f * 33, ['palma_abierta', 'puño']);
  ok(v2.llamadas.some(Array.isArray), 'al cerrar una de las manos, la palma que queda abierta desplaza la vista');
}

// 13. Con las dos palmas, mover las manos un tercio del cuadro da una vuelta entera al objeto
{
  const visor = visorFalso();
  const controles = crearControles(visor);
  const dos = ['palma_abierta', 'palma_abierta'];
  const palmas = (x) => [aImagen(POSES.palma_abierta, 0.2, x - 0.15, 0.6), aImagen(POSES.palma_abierta, 0.2, x + 0.15, 0.6)];
  // De x = 0,67 a 0,33 en 20 cuadros (la mano va hacia la derecha del usuario)
  for (let f = 0; f <= 20; f++) controles('palma_abierta', palmas(0.67 - (f / 20) * 0.34), f * 33, dos);
  const vueltas = visor.giro / (2 * Math.PI);
  ok(vueltas > 0.95 && vueltas < 1.1, `un tercio del cuadro con las dos palmas = ${vueltas.toFixed(2)} vueltas`);
}

// 14. Un movimiento lento (menos que la zona muerta por cuadro) se acumula en lugar de perderse
{
  let ancla = { x: 0.5, y: 0.5 };
  let total = 0;
  for (let f = 1; f <= 40; f++) {
    const m = seguir({ x: 0.5 - f * 0.001, y: 0.5 }, ancla); // 0,001 por cuadro: la mitad de la zona muerta
    total += m.dx;
    ancla = m.ancla;
  }
  ok(Math.abs(total - 0.04) < 0.0025, `40 cuadros de 0,001 suman ${total.toFixed(4)} (se esperaba 0,04)`);
  // El temblor alrededor de un punto no acumula nada
  ancla = { x: 0.5, y: 0.5 };
  total = 0;
  for (let f = 0; f < 60; f++) {
    const m = seguir({ x: 0.5 + (f % 2 ? 0.0015 : -0.0015), y: 0.5 }, ancla);
    total += m.dx;
    ancla = m.ancla;
  }
  ok(total === 0, `el temblor de ±0,0015 no mueve nada (${total})`);
}

// 15. Los recuadros que se arrastran no quedan fuera de la pantalla
{
  const pantalla = { anchoPantalla: 1600, altoPantalla: 900 };
  const caja = { ancho: 300, alto: 280 };
  const igual = (a, b) => a.x === b.x && a.y === b.y;
  ok(igual(dentroDePantalla({ x: 500, y: 300 }, caja, pantalla), { x: 500, y: 300 }), 'adentro queda donde se soltó');
  ok(igual(dentroDePantalla({ x: 1500, y: 850 }, caja, pantalla), { x: 1292, y: 612 }), 'pasado el borde de abajo a la derecha, vuelve adentro');
  ok(igual(dentroDePantalla({ x: -80, y: -40 }, caja, pantalla), { x: 8, y: 8 }), 'pasado el borde de arriba a la izquierda, vuelve adentro');
  ok(igual(dentroDePantalla({ x: 200, y: 200 }, { ancho: 2000, alto: 1200 }, pantalla), { x: 8, y: 8 }),
    'una caja más grande que la pantalla queda pegada arriba a la izquierda');
}

// 16. Los controles devuelven lo que aplicaron de verdad (el visor lo muestra: "Dos palmas · rotar")
{
  const visor = visorFalso();
  const controles = crearControles(visor);
  const palma = (x) => aImagen(POSES.palma_abierta, 0.2, x, 0.6);
  ok(controles('palma_abierta', [palma(0.5)], 0, ['palma_abierta']) === 'palma_abierta', 'una palma: devuelve palma_abierta');
  // De una palma a dos hacen falta 3 cuadros seguidos (histéresis)
  const dos = [33, 66, 99].map((t) => controles('palma_abierta', [palma(0.3), palma(0.7)], t, ['palma_abierta', 'palma_abierta']));
  ok(dos.join() === 'palma_abierta,palma_abierta,dos_palmas', `dos palmas: devuelve dos_palmas al tercer cuadro (${dos.join(', ')})`);
  ok(controles('escala_dos_manos', [palma(0.3)], 132) === 'ninguno', 'escalar sin las dos manos: devuelve ninguno');
  visor.piezaBajo = null;
  ok(controles('señalar', [aImagen(POSES['señalar'], 0.15, 0.5, 0.5)], 165) === 'señalar', 'señalar: devuelve señalar');
}

// 17. La sensibilidad del panel de ajustes multiplica el giro y el zoom
{
  const girar = (rotar) => {
    const visor = visorFalso();
    const controles = crearControles(visor, { ajustes: () => ({ ...AJUSTES_POR_DEFECTO, rotar }) });
    const palmas = (x) => [aImagen(POSES.palma_abierta, 0.2, x - 0.15, 0.6), aImagen(POSES.palma_abierta, 0.2, x + 0.15, 0.6)];
    for (let f = 0; f <= 10; f++) controles('palma_abierta', palmas(0.6 - f * 0.01), f * 33, ['palma_abierta', 'palma_abierta']);
    return visor.giro;
  };
  const [normal, doble] = [girar(1), girar(2)];
  ok(Math.abs(doble / normal - 2) < 1e-6, `con sensibilidad 2× gira el doble (${normal.toFixed(2)} → ${doble.toFixed(2)} rad)`);

  const escalar = (sens) => {
    let escala = 1;
    const visor = { ...visorFalso(), escala: () => 1, escalarA: (e) => { escala = e; } };
    const controles = crearControles(visor, { ajustes: () => ({ ...AJUSTES_POR_DEFECTO, escalar: sens }) });
    const manos = (sep) => [aImagen(POSES.pellizco, 0.12, 0.5 - sep, 0.7), aImagen(POSES.pellizco, 0.12, 0.5 + sep, 0.7)];
    controles('escala_dos_manos', manos(0.1), 0);
    controles('escala_dos_manos', manos(0.2), 33);
    return escala;
  };
  ok(Math.abs(escalar(1) - 2) < 0.05 && Math.abs(escalar(2) - 4) < 0.2,
    `separar las manos al doble: zoom ${escalar(1).toFixed(2)}× normal y ${escalar(2).toFixed(2)}× con sensibilidad 2×`);
}

// 18. Inercia: soltar las palmas girando rápido deja el objeto girando; quietas o sin inercia, no
{
  const palmas = (x) => [aImagen(POSES.palma_abierta, 0.2, x - 0.15, 0.6), aImagen(POSES.palma_abierta, 0.2, x + 0.15, 0.6)];
  const dos = ['palma_abierta', 'palma_abierta'];
  const probar = ({ inercia = true, quietoAlFinal = false } = {}) => {
    const visor = visorFalso();
    const controles = crearControles(visor, { ajustes: () => ({ ...AJUSTES_POR_DEFECTO, inercia }) });
    let t = 0;
    for (let f = 0; f <= 8; f++, t += 33) controles('palma_abierta', palmas(0.6 - f * 0.02), t, dos);
    if (quietoAlFinal) for (let f = 0; f < 10; f++, t += 33) controles('palma_abierta', palmas(0.44), t, dos);
    controles('ninguno', [], t);
    return visor.impulso;
  };
  const impulso = probar();
  ok(impulso && impulso[0] > 1.5 && impulso[1] === 0, `al soltar girando sigue girando a los costados (${impulso?.[0].toFixed(1)} rad/s)`);
  ok(impulso[0] <= 6 * Math.PI, 'la inercia tiene un máximo (tres vueltas por segundo)');
  ok(probar({ quietoAlFinal: true }) === null, 'si las manos estaban quietas al soltar, no hay inercia');
  ok(probar({ inercia: false }) === null, 'con la inercia apagada en ajustes, no sigue girando');
}

// 19. Los ajustes guardados se validan: fuera de rango o de otro tipo, vuelven a un valor válido
{
  const a = normalizarAjustes({ rotar: 10, desplazar: -1, escalar: 'x', umbral: 0.99, inercia: 'si', presentacion: { camara: true } });
  ok(a.rotar === 3 && a.desplazar === 0.3 && a.escalar === 1 && a.umbral === 0.95,
    `valores fuera de rango se limitan (rotar ${a.rotar}, desplazar ${a.desplazar}, escalar ${a.escalar}, umbral ${a.umbral})`);
  ok(a.inercia === true && a.presentacion.camara === true && a.presentacion.ia === true, 'los que no son del tipo correcto quedan por defecto');
  ok(JSON.stringify(normalizarAjustes(null)) === JSON.stringify(AJUSTES_POR_DEFECTO), 'sin nada guardado: los valores por defecto');
  ok(normalizarAjustes(null) !== AJUSTES_POR_DEFECTO && normalizarAjustes(null).presentacion !== AJUSTES_POR_DEFECTO.presentacion,
    'devuelve una copia: cambiarla no toca los valores por defecto');
}

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
