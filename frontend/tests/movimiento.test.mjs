// Movimiento de las piezas (frontend/js/movimiento.js) con los datos reales de los motores del catálogo.
// Uso (desde la raíz):  node frontend/tests/movimiento.test.mjs
import { readFileSync } from 'node:fs';
import {
  PERIODO, componer, crearMarcha, cuaternionEje, movimientoPropio, movimientos, rotar,
} from '../js/movimiento.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };
const aplicar = ({ q, t }, p) => rotar(q, p).map((v, k) => v + t[k]);
const distancia = (a, b) => Math.hypot(...a.map((v, k) => v - b[k]));

/** Piezas ({ id, movimiento }) y velocidad de un .glb generado. */
function leerGlb(ruta) {
  const datos = readFileSync(new URL(`../models/motores/${ruta}`, import.meta.url));
  const largo = datos.readUInt32LE(12);
  const gltf = JSON.parse(datos.subarray(20, 20 + largo).toString('utf8'));
  return {
    piezas: gltf.nodes.map((n) => ({ id: n.extras?.pieza ?? n.name, movimiento: n.extras?.movimiento })),
    velocidad: gltf.scenes[0].extras?.velocidad,
  };
}

// ---- rotaciones
const q = cuaternionEje([1, 0, 0], Math.PI / 2);
ok(distancia(rotar(q, [0, 1, 0]), [0, 0, 1]) < 1e-9, 'girar 90° sobre X lleva Y a Z (regla de la mano derecha, como el generador)');
const doble = componer({ q, t: [0, 0, 0] }, { q, t: [0, 0, 0] });
ok(distancia(rotar(doble.q, [0, 1, 0]), [0, -1, 0]) < 1e-9, 'componer dos giros de 90° da 180°');

// ---- motores: la biela sigue unida al pistón y al muñón del cigüeñal en toda la vuelta
const MOTORES = {
  'en línea de 4': 'combustion_4_linea_90p0_8p6_8p6_dohc_4_no_correa_nafta.glb',
  'V8': 'combustion_8_v_90p0_8p6_8p6_dohc_4_no_correa_nafta.glb',
  'bóxer de 6': 'combustion_6_boxer_90p0_8p6_8p6_dohc_4_si_correa_nafta.glb',
  'monocilíndrico': 'combustion_1_linea_90p0_8p6_8p6_sohc_2_no_correa_nafta.glb',
};
for (const [nombre, archivo] of Object.entries(MOTORES)) {
  const { piezas, velocidad } = leerGlb(archivo);
  ok(velocidad > 0, `${nombre}: el .glb declara la velocidad de la marcha`);
  const cigueñal = piezas.find((p) => p.id === 'ciguenal');
  const bielas = piezas.filter((p) => p.movimiento?.tipo === 'biela');
  ok(cigueñal?.movimiento?.tipo === 'giro' && bielas.length > 0, `${nombre}: el cigüeñal gira y hay ${bielas.length} biela(s)`);
  let peorArriba = 0;
  let peorAbajo = 0;
  let masAlto = -Infinity;
  let masBajo = Infinity;
  for (let k = 0; k < 720; k++) {
    const theta = (k / 720) * PERIODO;
    const movs = movimientos(piezas, theta);
    for (const biela of bielas) {
      const m = biela.movimiento;
      const n = biela.id.split('_').pop();
      const u = m.u;
      const perno0 = [m.x, u[1] * m.s0, u[2] * m.s0];
      const munon0 = [m.x, m.r * Math.cos(m.fase), m.r * Math.sin(m.fase)];
      const piston = movs.get(`piston_${n}`);
      peorArriba = Math.max(peorArriba, distancia(aplicar(movs.get(biela.id), perno0), aplicar(piston, perno0)));
      peorAbajo = Math.max(peorAbajo, distancia(aplicar(movs.get(biela.id), munon0), aplicar(movs.get('ciguenal'), munon0)));
      // posición del pistón a lo largo de su cilindro (lo que se movió desde el dibujo, proyectado en u)
      const d = piston.t[1] * u[1] + piston.t[2] * u[2];
      if (n === bielas[0].id.split('_').pop()) { masAlto = Math.max(masAlto, d); masBajo = Math.min(masBajo, d); }
    }
  }
  // los datos van redondeados a 5 decimales: hasta un micrómetro de diferencia es redondeo
  ok(peorArriba < 1e-4, `${nombre}: el extremo de arriba de cada biela no se separa del pistón (máx. ${peorArriba.toExponential(1)} cm)`);
  ok(peorAbajo < 1e-4, `${nombre}: el de abajo no se separa del muñón del cigüeñal (máx. ${peorAbajo.toExponential(1)} cm)`);
  const carrera = 2 * bielas[0].movimiento.r;
  const recorrido = masAlto - masBajo;
  ok(Math.abs(recorrido - carrera) < 0.01 * carrera, `${nombre}: el pistón recorre la carrera, de punto muerto a punto muerto (${recorrido.toFixed(2)} de ${carrera} cm)`);
  const quieto = movimientos(piezas, 0);
  ok([...quieto.values()].every(({ t }) => Math.hypot(...t) < 1e-4), `${nombre}: en θ = 0 todo está como se dibujó`);
}

// ---- levas a media vuelta, eléctrico, radial
const { piezas: cuatro } = leerGlb(MOTORES['en línea de 4']);
const leva = cuatro.find((p) => p.id.startsWith('arbol_levas'));
ok(leva.movimiento.relacion === 0.5, 'los árboles de levas giran a la mitad que el cigüeñal');
const { piezas: electrico } = leerGlb('electrico_induccion_4p0_brida.glb');
const giran = electrico.filter((p) => p.movimiento).map((p) => p.id).sort();
ok(['eje', 'rotor', 'ventilador'].every((id) => giran.includes(id)) && !giran.includes('estator') && !giran.includes('carcasa'),
  `eléctrico: giran ${giran.join(', ')}; el estator y la carcasa quedan quietos`);
const { piezas: radial } = leerGlb('combustion_9_radial_90p0_10p0_11p0_sohc_2_no_correa_nafta.glb');
const movRadial = movimientos(radial, 1.3);
ok(radial.filter((p) => p.id.startsWith('piston_')).every((p) => movRadial.has(p.id)), 'radial: se mueven todos los pistones');

// ---- articulaciones: oscilar y seguir al padre
const brazo = { id: 'brazo', movimiento: { tipo: 'oscilar', eje: [1, 0, 0], centro: [0, 10, 0], amplitud: 0.5, fase: 0 } };
const antebrazo = { id: 'antebrazo', movimiento: { tipo: 'oscilar', eje: [1, 0, 0], centro: [0, 5, 0], amplitud: 0, padre: 'brazo' } };
const m = movimientos([antebrazo, brazo], Math.PI / 2);
const codo = [0, 5, 0];
ok(distancia(aplicar(m.get('antebrazo'), codo), aplicar(m.get('brazo'), codo)) < 1e-9,
  'el antebrazo acompaña al brazo: el codo queda en el mismo lugar para los dos');
ok(distancia(aplicar(movimientoPropio(brazo.movimiento, 1.2, 0), [0, 0, 0]), [0, 0, 0]) < 1e-9,
  'con intensidad 0 (apagado) la articulación está como se dibujó');
const rodilla = { tipo: 'oscilar', eje: [1, 0, 0], base: 0.3, amplitud: 0.3, fase: -Math.PI / 2 };
ok(Array.from({ length: 36 }, (_, k) => k / 36 * 2 * Math.PI).every((th) => {
  const v = rotar(movimientoPropio(rodilla, th).q, [0, -1, 0]);
  return v[2] <= 1e-9;
}), 'la rodilla solo se dobla hacia atrás (base + amplitud·sin nunca cambia de signo)');
const ciclo = movimientos([{ id: 'a', movimiento: { tipo: 'oscilar', amplitud: 0.2, padre: 'b' } },
  { id: 'b', movimiento: { tipo: 'oscilar', amplitud: 0.2, padre: 'a' } }], 1);
ok(ciclo.size === 2, 'un ciclo de padres no cuelga el cálculo');

// ---- exoesqueleto: camina sin que se desarmen las articulaciones
const exo = (() => {
  const datos = readFileSync(new URL('../models/exoesqueleto/exoesqueleto_industrial.glb', import.meta.url));
  const largo = datos.readUInt32LE(12);
  const gltf = JSON.parse(datos.subarray(20, 20 + largo).toString('utf8'));
  return gltf.nodes.map((n) => ({ id: n.extras?.pieza ?? n.name, movimiento: n.extras?.movimiento }));
})();
const UNIONES = [['muslo', 'pierna'], ['pierna', 'bota'], ['brazo', 'antebrazo'], ['antebrazo', 'guante']];
let peorUnion = 0;
let maxCadera = 0;
for (let k = 0; k < 48; k++) {
  const movs = movimientos(exo, (k / 48) * 2 * Math.PI, 1);
  for (const lado of ['izquierdo', 'derecho']) {
    for (const [arriba, abajo] of UNIONES) {
      const centro = exo.find((p) => p.id === `${abajo}_${lado}`).movimiento.centro;
      peorUnion = Math.max(peorUnion, distancia(aplicar(movs.get(`${arriba}_${lado}`), centro), aplicar(movs.get(`${abajo}_${lado}`), centro)));
    }
  }
  const izq = movs.get('muslo_izquierdo');
  maxCadera = Math.max(maxCadera, 2 * Math.asin(Math.min(1, Math.hypot(...izq.q.slice(0, 3)))));
}
ok(peorUnion < 1e-4, `exoesqueleto: rodillas, tobillos, codos y muñecas no se separan al caminar (máx. ${peorUnion.toExponential(1)} cm)`);
ok(maxCadera > 0.3 && maxCadera < 0.4, `exoesqueleto: la cadera oscila ±${(maxCadera * 180 / Math.PI).toFixed(0)}°`);
const enContrafase = movimientos(exo, Math.PI / 2, 1);
const zPie = (lado) => aplicar(enContrafase.get(`bota_${lado}`), exo.find((p) => p.id === `bota_${lado}`).movimiento.centro)[2];
ok(Math.sign(zPie('izquierdo')) !== Math.sign(zPie('derecho')), 'exoesqueleto: un pie va adelante y el otro atrás');
ok(!exo.find((p) => p.id === 'columna').movimiento, 'exoesqueleto: la estructura dorsal queda quieta');

// ---- reloj de la marcha
const marcha = crearMarcha(2 * Math.PI);
marcha.encender();
for (let k = 0; k < 13; k++) marcha.avanzar(0.1);
ok(marcha.theta > 0 && marcha.encendida, 'encendida, el ángulo avanza');
marcha.apagar();
ok(marcha.moviendose && !marcha.encendida, 'al apagar sigue hasta terminar el ciclo');
for (let k = 0; k < 40; k++) marcha.avanzar(0.1);
ok(marcha.theta === 0 && !marcha.moviendose, 'y se detiene con las piezas como se dibujaron');
const suave = crearMarcha(2 * Math.PI);
suave.encender();
suave.avanzar(0.1);
ok(suave.intensidad > 0 && suave.intensidad < 1, 'las articulaciones arrancan de a poco');
for (let k = 0; k < 10; k++) suave.avanzar(0.1);
ok(suave.intensidad === 1, 'y llegan a todo su movimiento');
ok(crearMarcha(null).avanzar(1) === 0, 'sin velocidad no se mueve');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
