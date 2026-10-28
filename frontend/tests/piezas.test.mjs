// Pruebas del desmontaje por piezas (frontend/js/piezas.js).
// Uso (desde la raíz):  node frontend/tests/piezas.test.mjs
import { desplazamiento, prepararPiezas, siguienteParaDesmontar, ultimaDesmontada } from '../js/piezas.js';

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

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
