// Pruebas del desmontaje por piezas (frontend/js/piezas.js).
// Uso (desde la raíz):  node frontend/tests/piezas.test.mjs
import {
  crearSeleccionPorPermanencia, desplazamiento, encuadreAislada, nombreMaterial, planificarDesarme, prepararPiezas,
  siguienteParaDesmontar,
  textoMedidas, ultimaDesmontada,
} from '../js/piezas.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

// Una válvula simplificada: cuerpo fijo, volante y vástago hacia arriba, bridas a los costados
const valvula = [
  { id: 'cuerpo', nombre: 'Cuerpo', centro: [0, 0, 0], volumen: 8 },
  { id: 'vastago', nombre: 'Vástago', orden: 2, direccion: [0, 1, 0], centro: [0, 1.0, 0], volumen: 0.1 },
  { id: 'volante', nombre: 'Volante', orden: 1, direccion: [0, 1, 0], centro: [0, 1.3, 0], volumen: 0.5 },
  { id: 'brida', nombre: 'Brida', orden: 3, direccion: [-2, 0, 0], centro: [-0.6, 0, 0], volumen: 1 },
];
const piezas = prepararPiezas(valvula, [0, 0, 0], 3);

ok(piezas.map((p) => p.id).join() === 'volante,vastago,brida,cuerpo', `orden de desmontaje: ${piezas.map((p) => p.id)}`);
ok(piezas.at(-1).fija && !piezas[0].fija, 'la pieza sin orden queda fija y al final');
ok(JSON.stringify(piezas[2].dir) === JSON.stringify([-1, 0, 0]), 'la dirección se normaliza');
ok(piezas[0].dist > piezas[1].dist, 'el volante (más arriba) se aleja más que el vástago: no se superponen');

// Desmontar y montar de a una
const objetivos = piezas.map(() => 0);
let i = siguienteParaDesmontar(piezas, objetivos);
ok(piezas[i].id === 'volante', 'primero sale el volante');
objetivos[i] = 1;
i = siguienteParaDesmontar(piezas, objetivos);
ok(piezas[i].id === 'vastago', 'después el vástago');
objetivos[i] = 1;
ok(piezas[ultimaDesmontada(piezas, objetivos)].id === 'vastago', 'se vuelve a montar primero la última que salió');
objetivos[2] = 1;
ok(siguienteParaDesmontar(piezas, objetivos) === -1, 'con todas afuera no queda ninguna para desmontar (la fija no cuenta)');
ok(ultimaDesmontada(piezas, piezas.map(() => 0)) === -1, 'armado: no hay nada para montar');

const d = desplazamiento(piezas[0], 0.5);
ok(Math.abs(d[1] - piezas[0].dist * 0.5) < 1e-9 && d[0] === 0, 'el desplazamiento sigue la dirección y la fracción');

// Sin metadatos de orden (por ejemplo, un modelo generado): queda fija la más grande y salen desde afuera
const sinOrden = prepararPiezas([
  { id: 'chica_cerca', centro: [0.2, 0, 0], volumen: 1 },
  { id: 'grande', centro: [0, 0, 0], volumen: 10 },
  { id: 'chica_lejos', centro: [0, 2, 0], volumen: 1 },
], [0, 0, 0], 3);
ok(sinOrden.map((p) => p.id).join() === 'chica_lejos,chica_cerca,grande', `sin orden: ${sinOrden.map((p) => p.id)}`);
ok(JSON.stringify(sinOrden[0].dir) === JSON.stringify([0, 1, 0]), 'sin dirección: se aleja desde el centro');

// En la misma dirección, la pieza que sale antes queda más afuera (bulones por fuera de la brida)
const brida = prepararPiezas([
  { id: 'cuerpo', centro: [0, 0, 0], volumen: 8 },
  { id: 'bulones', orden: 1, direccion: [-1, 0, 0], centro: [-0.6, 0, 0], volumen: 0.1 },
  { id: 'brida', orden: 2, direccion: [-1, 0, 0], centro: [-0.6, 0, 0], volumen: 1 },
], [0, 0, 0], 3);
ok(brida[0].dist > brida[1].dist, `bulones (${brida[0].dist.toFixed(2)}) más afuera que la brida (${brida[1].dist.toFixed(2)})`);

// Aislar una pieza: queda centrada (el contenedor se corre al revés de su centro) y más grande cuanto más chica es
const chica = encuadreAislada([1, 2, 0], 0.1, 1, 1);
const grande = encuadreAislada([1, 2, 0], 0.5, 1, 1);
ok(chica.escala > grande.escala, 'una pieza chica se agranda más que una grande');
ok(Math.abs(chica.posicion[0] + chica.escala) < 1e-9 && Math.abs(chica.posicion[1] + 2 * chica.escala) < 1e-9,
  'el centro de la pieza queda en el origen');
ok(Number.isFinite(encuadreAislada([0, 0, 0], 0, 1, 1).escala), 'una pieza sin tamaño no rompe el encuadre');

ok(nombreMaterial('acero') === 'Acero' && nombreMaterial('laton_c08040') === 'Latón', 'materiales conocidos, con o sin color');
ok(nombreMaterial('volante') === null, 'un nombre que no es material no se muestra');
ok(textoMedidas([12.5, 4, 0.5], 'cm') === '13 × 4,0 × 0,5 cm', `medidas con unidad: ${textoMedidas([12.5, 4, 0.5], 'cm')}`);
ok(textoMedidas([2, 1, 0.5]).startsWith('100 × 50 × 25'), `sin unidad, proporciones: ${textoMedidas([2, 1, 0.5])}`);

// Selección por permanencia: hay que quedarse 800 ms sobre la misma pieza, y elige una sola vez
const permanencia = crearSeleccionPorPermanencia(800);
ok(permanencia('volante', 0).elegida === null, 'al empezar a apuntar no elige');
ok(Math.abs(permanencia('volante', 400).progreso - 0.5) < 1e-9, 'a mitad de tiempo, progreso 0,5');
ok(permanencia('volante', 800).elegida === 'volante', 'a los 800 ms elige la pieza');
ok(permanencia('volante', 1200).elegida === null, 'no la vuelve a elegir mientras sigue apuntando');
ok(permanencia('vastago', 1300).progreso === 0, 'apuntar a otra pieza reinicia la cuenta');
ok(permanencia(null, 1400).progreso === 0 && permanencia('vastago', 1500).elegida === null, 'apuntar al vacío también');

// A* para el orden de desarme (modelos sin orden): una base, una placa encima, una tapa sobre la placa y un
// tornillo metido en la base por el costado. La tapa tiene que salir antes que la placa, y el tornillo por su lado.
const apilado = [
  { id: 'base', centro: [0, 0, 0], tamano: [2, 1, 2] },
  { id: 'placa', centro: [0, 0.6, 0], tamano: [1.8, 0.2, 1.8] },
  { id: 'tapa', centro: [0, 0.8, 0], tamano: [0.5, 0.2, 0.5] },
  { id: 'tornillo', centro: [0.9, 0, 0], tamano: [0.6, 0.2, 0.2] },   // metido en la base, la cabeza afuera
];
const plan = planificarDesarme(apilado, 0);
const orden = plan.pasos.map((p) => apilado[p.pieza].id);
ok(orden.indexOf('tapa') < orden.indexOf('placa'), `A*: la tapa sale antes que la placa (${orden})`);
ok(plan.pasos.every((p) => !p.trabada), 'A*: ninguna pieza queda trabada');
const tornillo = plan.pasos.find((p) => apilado[p.pieza].id === 'tornillo');
ok(tornillo.dir.join() === '1,0,0', `A*: el tornillo sale por el costado (${tornillo.dir})`);
const placa = plan.pasos.find((p) => apilado[p.pieza].id === 'placa');
ok(placa.dir.join() === '0,1,0' || Math.abs(placa.dir[0]) + Math.abs(placa.dir[2]) === 1, 'A*: la placa sale libre');

// Un eje dentro de una carcasa cerrada: no puede salir sin chocar, pero el plan igual termina (marcada trabada)
const trabado = planificarDesarme([
  { id: 'carcasa', centro: [0, 0, 0], tamano: [2, 2, 2] },
  { id: 'eje', centro: [0, 0, 0], tamano: [0.3, 1.5, 0.3] },
], 0);
ok(trabado.pasos.length === 1 && trabado.pasos[0].trabada, 'A*: una pieza encerrada sale igual, marcada como trabada');

// prepararPiezas usa A* cuando se conocen las cajas y no hay orden
const conCajas = prepararPiezas(apilado.map((p) => ({ ...p, volumen: p.tamano.reduce((a, b) => a * b) })), [0, 0.3, 0], 3);
const ids = conCajas.map((p) => p.id);
ok(ids.at(-1) === 'base' && ids.indexOf('tapa') < ids.indexOf('placa'), `prepararPiezas con A*: ${ids}`);

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
