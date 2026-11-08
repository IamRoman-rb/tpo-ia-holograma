// La inferencia del MLP en JavaScript (frontend/js/mlp.js) da lo mismo que en Python (scikit-learn o PyTorch).
// Los casos los genera el script de entrenamiento con filas de PRUEBA y las probabilidades de Python.
// Uso (desde la raíz):  node frontend/tests/mlp.test.mjs
import { readFileSync } from 'node:fs';
import { crearClasificadorMlp, crearPredictor, crearPropagacion, importanciaPuntos } from '../js/mlp.js';
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
const capas = modelo.version === 3 ? modelo.bloques.length + 2 : modelo.capas.length;   // v3: entrada, bloques, salida
ok(tamanos.length === capas + 1 && tamanos[0] === 63 && tamanos.at(-1) === modelo.clases.length,
  `devuelve las activaciones de cada capa (${tamanos.join(' → ')})`);
ok(Math.abs(salida.probabilidades.reduce((s, v) => s + v, 0) - 1) < 1e-9, 'y las probabilidades de todas las clases (suman 1)');

// Los pesos pueden venir en float32 base64 (versión 2) o como listas (versión 1): los dos dan lo mismo
const aListas = (m) => {
  if (Array.isArray(m)) return m;
  const plano = Array.from(new Float32Array(new Uint8Array(Buffer.from(m.float32, 'base64')).buffer));
  const [filas, cols] = m.forma;
  return cols ? Array.from({ length: filas }, (_, i) => plano.slice(i * cols, (i + 1) * cols)) : plano;
};
if (modelo.capas) {   // versiones 1 y 2 (la 3 solo se guarda en base64)
  const enListas = { ...modelo, capas: modelo.capas.map(({ W, b }) => ({ W: aListas(W), b: aListas(b) })) };
  const x0 = casos[0].x;
  const dif = Math.max(...crearPredictor(enListas)(x0).map((v, i) => Math.abs(v - crearPredictor(modelo)(x0)[i])));
  ok(dif < 1e-6, `los pesos en base64 y como listas dan las mismas probabilidades (diferencia ${dif.toExponential(1)})`);
}

// El clasificador temporal acepta el MLP en lugar de las reglas
const fijo = () => ({ gesto: 'victoria', probabilidad: 0.9 });
const clasificar = crearClasificador({ mlp: fijo });
let r;
for (let i = 0; i < 6; i++) r = clasificar([mano], 1);
ok(r.gesto === 'victoria' && r.confianza === 1, 'con el MLP, la ventana temporal funciona igual');

// Qué puntos pesan en la decisión (red.html): con manos reales de HaGRID que la red no vio. En un pellizco tiene
// que pesar el pulgar o el índice; en todas, la muñeca vale 0 (es el origen) y el máximo es 1.
{
  const propagar = crearPropagacion(modelo);
  const { manos } = JSON.parse(readFileSync(new URL('../ml/manos_demo.json', import.meta.url), 'utf8'));
  const DEDOS_PELLIZCO = new Set([2, 3, 4, 6, 7, 8]);   // pulgar e índice (sin los nudillos de la palma)
  let pellizcosBien = 0;
  let pellizcos = 0;
  for (const { clase, puntos } of manos) {
    const k = modelo.clases.indexOf(clase);
    if (k < 0) continue;
    const imp = importanciaPuntos(propagar, puntos, k);
    ok(imp.length === 21 && imp[0] === 0 && Math.max(...imp) === 1 && imp.every((v) => v >= 0 && v <= 1),
      `importancia de 21 puntos entre 0 y 1 (${clase})`);
    if (clase === 'pellizco') {
      pellizcos++;
      const masPesado = imp.indexOf(1);
      if (DEDOS_PELLIZCO.has(masPesado)) pellizcosBien++;
    }
  }
  ok(pellizcos > 0 && pellizcosBien >= Math.ceil(pellizcos / 2),
    `en ${pellizcosBien} de ${pellizcos} pellizcos lo que más pesa es el pulgar o el índice`);
  // No modifica el vector de entrada
  const v = manos[0].puntos.slice();
  importanciaPuntos(propagar, v, 0);
  ok(v.every((x, i) => x === manos[0].puntos[i]), 'calcular la importancia no cambia los puntos de la mano');
}

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
