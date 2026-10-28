// La inferencia del MLP en JavaScript (frontend/js/mlp.js) da lo mismo que scikit-learn.
// Los casos los genera ml/entrenar_mlp.py con filas de PRUEBA y las probabilidades de Python.
// Uso (desde la raíz):  node frontend/tests/mlp.test.mjs
import { readFileSync } from 'node:fs';
import { crearClasificadorMlp, crearPredictor } from '../js/mlp.js';
import { crearClasificador } from '../js/gestures.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

const modelo = JSON.parse(readFileSync(new URL('../ml/mlp_gestos.json', import.meta.url), 'utf8'));
const { clases, casos } = JSON.parse(readFileSync(new URL('./datos/mlp_casos.json', import.meta.url), 'utf8'));

ok(JSON.stringify(modelo.clases) === JSON.stringify(clases), 'el modelo y los casos usan las mismas clases');
const predecir = crearPredictor(modelo);
let peor = 0;
for (const { x, proba } of casos) {
  const p = predecir(x);
  peor = Math.max(peor, ...p.map((v, i) => Math.abs(v - proba[i])));
}
// Los pesos se exportan con 6 decimales: la diferencia tiene que ser mínima
ok(peor < 1e-3, `mismas probabilidades que scikit-learn en ${casos.length} casos (diferencia máxima ${peor.toExponential(1)})`);

// Por debajo del umbral de confianza, la mano cuenta como "ninguno"
const dudoso = { ...modelo, umbral_confianza: 1.01 };
const mano = Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + 0.01 * i, y: 0.5 - 0.02 * i, z: 0 }));
ok(crearClasificadorMlp(dudoso)(mano, 1).gesto === 'ninguno', 'si no llega al umbral, devuelve ninguno');

// El panel de actividad dibuja las activaciones de cada capa: entrada 63, ocultas 128 y 64, salida 8
const salida = crearClasificadorMlp(modelo)(mano, 1);
ok(JSON.stringify(salida.activaciones.map((a) => a.length)) === '[63,128,64,8]', 'devuelve las activaciones de cada capa');
ok(Math.abs(salida.probabilidades.reduce((s, v) => s + v, 0) - 1) < 1e-9, 'y las probabilidades de todas las clases (suman 1)');

// El clasificador temporal acepta el MLP en lugar de las reglas
const fijo = () => ({ gesto: 'victoria', probabilidad: 0.9 });
const clasificar = crearClasificador({ mlp: fijo });
let r;
for (let i = 0; i < 6; i++) r = clasificar([mano], 1);
ok(r.gesto === 'victoria' && r.confianza === 1, 'con el MLP, la ventana temporal funciona igual');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
