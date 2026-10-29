// La inferencia del MLP en JavaScript (frontend/js/mlp.js) da lo mismo que en Python (scikit-learn o PyTorch).
// Los casos los genera el script de entrenamiento con filas de PRUEBA y las probabilidades de Python.
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
ok(peor < 1e-3, `mismas probabilidades que en Python (entrenamiento) en ${casos.length} casos (diferencia máxima ${peor.toExponential(1)})`);

// Por debajo del umbral de confianza, la mano cuenta como "ninguno"
const dudoso = { ...modelo, umbral_confianza: 1.01 };
const mano = Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + 0.01 * i, y: 0.5 - 0.02 * i, z: 0 }));
ok(crearClasificadorMlp(dudoso)(mano, 1).gesto === 'ninguno', 'si no llega al umbral, devuelve ninguno');

// El panel de actividad dibuja las activaciones de cada capa: entrada 63, las ocultas y la salida (una por clase)
const salida = crearClasificadorMlp(modelo)(mano, 1);
const tamanos = salida.activaciones.map((a) => a.length);
ok(tamanos.length === modelo.capas.length + 1 && tamanos[0] === 63 && tamanos.at(-1) === modelo.clases.length,
  `devuelve las activaciones de cada capa (${tamanos.join(' → ')})`);
ok(Math.abs(salida.probabilidades.reduce((s, v) => s + v, 0) - 1) < 1e-9, 'y las probabilidades de todas las clases (suman 1)');

// Los pesos pueden venir en float32 base64 (versión 2) o como listas (versión 1): los dos dan lo mismo
const aListas = (m) => {
  if (Array.isArray(m)) return m;
  const plano = Array.from(new Float32Array(new Uint8Array(Buffer.from(m.float32, 'base64')).buffer));
  const [filas, cols] = m.forma;
  return cols ? Array.from({ length: filas }, (_, i) => plano.slice(i * cols, (i + 1) * cols)) : plano;
};
const enListas = { ...modelo, capas: modelo.capas.map(({ W, b }) => ({ W: aListas(W), b: aListas(b) })) };
const x0 = casos[0].x;
const dif = Math.max(...crearPredictor(enListas)(x0).map((v, i) => Math.abs(v - crearPredictor(modelo)(x0)[i])));
ok(dif < 1e-6, `los pesos en base64 y como listas dan las mismas probabilidades (diferencia ${dif.toExponential(1)})`);

// El clasificador temporal acepta el MLP en lugar de las reglas
const fijo = () => ({ gesto: 'victoria', probabilidad: 0.9 });
const clasificar = crearClasificador({ mlp: fijo });
let r;
for (let i = 0; i < 6; i++) r = clasificar([mano], 1);
ok(r.gesto === 'victoria' && r.confianza === 1, 'con el MLP, la ventana temporal funciona igual');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
