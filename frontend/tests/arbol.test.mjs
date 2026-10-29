// El árbol de decisión en JavaScript (frontend/js/arbol.js) da lo mismo que en Python (scikit-learn): los mismos
// rasgos de la mano y las mismas probabilidades. Los casos los genera ml/arboles/entrenar_arboles.py con filas
// de PRUEBA. Uso (desde la raíz):  node frontend/tests/arbol.test.mjs
import { readFileSync } from 'node:fs';
import { RASGOS, crearClasificadorArbol, rasgosArbol, recorrer, textoCamino } from '../js/arbol.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

const modelo = JSON.parse(readFileSync(new URL('../ml/arbol_gestos.json', import.meta.url), 'utf8'));
const { clases, casos } = JSON.parse(readFileSync(new URL('./datos/arbol_casos.json', import.meta.url), 'utf8'));
ok(JSON.stringify(modelo.clases) === JSON.stringify(clases), 'el modelo y los casos usan las mismas clases');
ok(JSON.stringify(modelo.rasgos) === JSON.stringify(RASGOS), 'los rasgos están en el mismo orden que en Python');

const aPuntos = (x) => Array.from({ length: 21 }, (_, i) => ({ x: x[3 * i], y: x[3 * i + 1], z: x[3 * i + 2] }));
let peorRasgo = 0;
let peorProba = 0;
for (const c of casos) {
  const r = rasgosArbol(aPuntos(c.x));
  peorRasgo = Math.max(peorRasgo, ...RASGOS.map((n, k) => Math.abs(r[n] - c.rasgos[k]) / Math.max(1, Math.abs(c.rasgos[k]))));
  const { probabilidades } = recorrer(modelo, r);
  peorProba = Math.max(peorProba, ...probabilidades.map((v, k) => Math.abs(v - c.proba[k])));
}
ok(peorRasgo < 1e-3, `mismos rasgos que en Python en ${casos.length} casos (diferencia relativa máxima ${peorRasgo.toExponential(1)})`);
ok(peorProba < 1e-3, `mismas probabilidades que scikit-learn (diferencia máxima ${peorProba.toExponential(1)})`);

// El camino se puede leer y termina en el gesto
const clasificar = crearClasificadorArbol(modelo);
const mano = aPuntos(casos[0].x);
const salida = clasificar(mano, 1);
ok(salida.camino.length >= 1 && salida.camino.length <= modelo.profundidad, `el camino tiene ${salida.camino.length} preguntas`);
const texto = textoCamino(salida.camino, salida.gesto);
ok(texto.endsWith(salida.gesto) && texto.includes('→'), `camino legible: ${texto.slice(0, 90)}…`);
ok(crearClasificadorArbol({ ...modelo, umbral_confianza: 1.01 })(mano, 1).gesto === 'ninguno',
  'si no llega al umbral, devuelve ninguno');

console.log(fallas ? `\n${fallas} fallas` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
