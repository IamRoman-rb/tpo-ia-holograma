// Pruebas de qué nodos de un .glb son piezas (nivelDePiezas y agruparPorCercania de frontend/js/piezas.js).
// Uso (desde la raíz):  node frontend/tests/despiece_glb.test.mjs
import {
  MAXIMO_PIEZAS, agruparPorCercania, centroTriangulos, componentesConexas, nivelDePiezas, nombresPorPosicion,
  hayQueSeccionar, prepararPiezas, seccionar,
} from '../js/piezas.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

// Árbol de prueba: { nombre, malla?, hijos? }
const nodo = (nombre, hijos = [], malla = false) => ({ nombre, hijos, malla });
const malla = (nombre, hijos = []) => nodo(nombre, hijos, true);
const piezas = (escena) => nivelDePiezas(escena, (n) => n.hijos, (n) => n.malla).map((n) => n.nombre);

// Como los de Sketchfab: raíz única y las mallas tres niveles más abajo (más una cámara, que no es pieza)
const sketchfab = nodo('escena', [
  nodo('Sketchfab_model', [
    nodo('RootNode', [
      nodo('camara'),
      nodo('Motor', [malla('bloque'), malla('culata'), nodo('carter', [malla('carter_malla')])]),
    ]),
  ]),
]);
ok(piezas(sketchfab).join() === 'bloque,culata,carter', `raíz única: baja hasta las piezas (${piezas(sketchfab)})`);

// Modelo propio: un nodo con malla por pieza, directamente bajo la escena. Quedan los mismos nodos, en el mismo orden.
const propio = nodo('escena', ['volante', 'vastago', 'bonete', 'cuerpo'].map((n) => malla(n)));
const elegidas = nivelDePiezas(propio, (n) => n.hijos, (n) => n.malla);
ok(elegidas.length === propio.hijos.length && elegidas.every((n, i) => n === propio.hijos[i]),
  `modelo propio: los hijos de la escena, sin cambios (${elegidas.map((n) => n.nombre)})`);

// Una malla suelta al lado del grupo que tiene todo lo demás: el grupo se abre y la suelta queda como pieza
const suelta = nodo('escena', [nodo('SketchUp', [
  malla('Material2'),
  nodo('instance_0', [nodo('Component_58', Array.from({ length: 19 }, (_, i) => malla(`parte_${i}`)))]),
])]);
const ps = piezas(suelta);
ok(ps.length === 20 && ps[0] === 'Material2' && ps[1] === 'parte_0', `un grupo con casi todas las mallas se abre (${ps.length} piezas)`);

// Un nodo con malla propia no se abre aunque tenga hijos: moverlo movería a los hijos
ok(piezas(nodo('escena', [malla('cuerpo', [malla('tapa'), malla('tornillo')])])).join() === 'cuerpo',
  'un nodo con malla y con hijos es una sola pieza');
ok(piezas(nodo('escena', [nodo('vacio')])).length === 0, 'sin mallas no hay piezas');

// Más de 80 piezas: se agrupan por cercanía en 80 como mucho
let semilla = 7;
const azar = () => { semilla = (semilla * 1103515245 + 12345) % 2147483648; return semilla / 2147483648; };
const montones = Array.from({ length: MAXIMO_PIEZAS }, (_, m) => [m * 10, (m % 7) * 10, (m % 3) * 10]);
const centros = Array.from({ length: 1400 }, (_, i) => montones[i % MAXIMO_PIEZAS].map((v) => v + azar()));
const grupos = agruparPorCercania(centros);
const todos = grupos.flat().sort((a, b) => a - b);
ok(grupos.length === MAXIMO_PIEZAS, `1.400 piezas en ${grupos.length} grupos`);
ok(todos.length === centros.length && todos.every((v, i) => v === i), 'cada pieza queda en un solo grupo');
// 80 montones bien separados: las piezas de un mismo montón (a menos de 2 de distancia) quedan juntas
const montonDe = (i) => i % MAXIMO_PIEZAS;
ok(montones.every((_, m) => grupos.filter((g) => g.some((i) => montonDe(i) === m)).length === 1),
  'las piezas cercanas quedan en el mismo grupo');
ok(grupos.every((g, k) => k === 0 || g[0] > grupos[k - 1][0]), 'los grupos salen ordenados por su primera pieza');

const pocas = agruparPorCercania([[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
ok(JSON.stringify(pocas) === '[[0],[1],[2]]', 'con 80 o menos, cada pieza sola');
ok(agruparPorCercania(Array.from({ length: 90 }, () => [1, 1, 1])).length === 1, 'puntos repetidos: un solo grupo');

// Decenas de piezas sin dirección (salen desde el centro): no se escalonan, así el despiece no se va lejísimos
const radiales = prepararPiezas(Array.from({ length: 60 }, (_, i) => ({
  id: `p${i}`, centro: [1 + i * 0.01, 0, 0], volumen: i === 0 ? 10 : 1,
})), [0, 0, 0], 3);
const lejos = Math.max(...radiales.map((p) => p.dist));
ok(lejos < 3, `60 piezas desde el centro: la más lejana sale a ${lejos.toFixed(2)} (menos que el tamaño del modelo)`);

// ---------------------------------------------------------------- modelos de una sola malla
{
// Un cubo como lo exporta un glTF: 24 vértices (4 por cara, repetidos en las aristas por las normales y la textura)
function cubo(c, lado = 1) {
  const h = lado / 2;
  const caras = [[1, 2, 0], [0, 2, 1], [0, 1, 2]];   // (eje u, eje v, normal): 6 caras con los dos signos
  const pos = [];
  const idx = [];
  for (const [a, b, n] of caras) {
    for (const s of [-1, 1]) {
      const base = pos.length / 3;
      for (const [u, v] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const p = [0, 0, 0];
        p[a] = u * h; p[b] = v * h; p[n] = s * h;
        pos.push(c[0] + p[0], c[1] + p[1], c[2] + p[2]);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  return { pos, idx };
}
function unir(...mallas) {
  const pos = [];
  const idx = [];
  for (const m of mallas) {
    const base = pos.length / 3;
    pos.push(...m.pos);
    idx.push(...m.idx.map((i) => i + base));
  }
  return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

const unCubo = unir(cubo([0, 0, 0]));
ok(componentesConexas(unCubo.pos, unCubo.idx).length === 1,
  'un cubo con vértices repetidos en las aristas es una sola parte (se unen por posición)');

const tres = unir(cubo([0, 0, 0], 2), cubo([5, 0, 0]), cubo([0, 5, 0], 0.5));
const partes = componentesConexas(tres.pos, tres.idx);
ok(partes.length === 3 && partes.every((p) => p.length === 12), `tres cubos sueltos en una malla: ${partes.length} partes de 12 triángulos`);
const centrosPartes = partes.map((p) => centroTriangulos(tres.pos, tres.idx, p).map((v) => Math.round(v * 100) / 100));
ok(centrosPartes.some((c) => c.join() === '5,0,0') && centrosPartes.some((c) => c.join() === '0,5,0'), `centros de las partes: ${JSON.stringify(centrosPartes)}`);

// Sin buffer de índices (cada triángulo con sus tres vértices): igual se unen
const plano = { pos: Float32Array.from([...tres.idx].flatMap((i) => [...tres.pos.slice(3 * i, 3 * i + 3)])) };
ok(componentesConexas(plano.pos, null).length === 3, 'sin índices: también tres partes');

// Una barra larga de una sola pieza (20 cubos pegados, que comparten caras): se corta en 6 secciones
const barra = unir(...Array.from({ length: 20 }, (_, i) => cubo([i, 0, 0])));
const todos = Array.from({ length: barra.idx.length / 3 }, (_, t) => t);
ok(componentesConexas(barra.pos, barra.idx).length === 1, 'la barra de cubos pegados es una sola parte');
const secciones = seccionar(barra.pos, barra.idx, todos, 6);
const repartidos = secciones.flat().sort((a, b) => a - b);
ok(secciones.length === 6, `la barra se corta en ${secciones.length} secciones`);
ok(repartidos.length === todos.length && repartidos.every((t, i) => t === i), 'cada triángulo queda en una sola sección');
const xs = secciones.map((s) => centroTriangulos(barra.pos, barra.idx, s)[0]).sort((a, b) => a - b);
ok(xs.every((x, i) => i === 0 || x - xs[i - 1] > 1.5), `las secciones son tramos a lo largo de la barra (x = ${xs.map((x) => x.toFixed(1))})`);
ok(JSON.stringify(seccionar(barra.pos, barra.idx, todos, 6)) === JSON.stringify(secciones), 'seccionar es determinístico');

ok(hayQueSeccionar([100, 3]) && hayQueSeccionar([900, 30, 30, 40]) && !hayQueSeccionar([300, 250, 250, 200]),
  'se corta la parte más grande si hay menos de 4 o si tiene más del 80 % de los triángulos');
const mixtos = nombresPorPosicion([[0, 2, 0], [0, -2, 0]], [0, 0, 0], 4, ['Pieza', 'Sección']);
ok(mixtos.join('|') === 'Pieza superior|Sección inferior', `prefijo por pieza: ${mixtos.join(', ')}`);
const nombres = nombresPorPosicion([[0, 2, 0], [0, -2, 0], [-2, 0, 0], [2, 0, 2], [0, 0, 0], [0, 2.1, 0]], [0, 0, 0], 4, 'Sección');
ok(nombres.join('|') === 'Sección superior 1|Sección inferior|Sección izquierda|Sección delantera derecha|Sección central|Sección superior 2',
  `nombres por posición: ${nombres.join(', ')}`);
}

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
