// Pruebas del reconocimiento de piezas en el navegador (frontend/js/reconocer_piezas.js).
// Uso (desde la raíz):  node frontend/tests/reconocer_piezas.test.mjs
// Las cuentas se comparan con las de Python (ml/piezas/rasgos.py y la red de ml/piezas/entrenar_piezas.py), con
// el motor de prueba que exporta el entrenamiento en tests/datos/piezas_fixture.json.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  autovalores3, clasesPermitidas, leerRed, muestrearPieza, nombrar, predecir, rasgosDeMuestras,
} from '../js/reconocer_piezas.js';
import { nombreGenerico } from '../js/piezas.js';
import { crearAzar } from '../js/holograma.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };
const aqui = dirname(fileURLToPath(import.meta.url));

// --- 1. Autovalores de una matriz simétrica 3×3
const M = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1]];
const { valores, vectores } = autovalores3(M);
const residuo = Math.max(...[0, 1, 2].map((k) => {
  const v = [vectores[0][k], vectores[1][k], vectores[2][k]];
  return Math.max(...[0, 1, 2].map((i) => Math.abs(M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2] - valores[k] * v[i])));
}));
ok(residuo < 1e-10, `M·v = λ·v para los tres autovectores (residuo ${residuo.toExponential(1)})`);
ok(Math.abs(valores.reduce((s, x) => s + x, 0) - 8) < 1e-10, 'la suma de los autovalores es la traza');

// --- 2. Muestreo: los puntos caen sobre la superficie, en coordenadas del modelo (con la matriz de la malla)
// Un cuadrado de 2 × 2 en z = 0, corrido 10 en x por la matriz (column-major, como three.js)
const cuadrado = { xyz: new Float32Array([0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  matriz: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1] };
const m = muestrearPieza([cuadrado], 500, crearAzar(3));
let adentro = true;
for (let i = 0; i < 500; i++) {
  const [x, y, z] = [m.puntos[i * 3], m.puntos[i * 3 + 1], m.puntos[i * 3 + 2]];
  if (x < 10 - 1e-9 || x > 12 + 1e-9 || y < -1e-9 || y > 2 + 1e-9 || Math.abs(z) > 1e-9) adentro = false;
}
ok(adentro, 'los puntos muestreados caen sobre el cuadrado, corrido por la matriz de la malla');
ok(Math.abs(m.area - 4) < 1e-9, `el área es la del cuadrado (${m.area})`);

// --- 3. Nombres que no dicen qué es la pieza (los que puede nombrar la red)
for (const n of ['Sección superior izquierda 2', 'Pieza central', 'Grupo 3 (12 partes)', 'Object_12', 'Cylinder.004',
  'polySurface37', 'mesh_0', 'Node 5']) ok(nombreGenerico(n), `«${n}» es un nombre genérico`);
for (const n of ['Piston', 'Carburetor_01', 'Bujía 2', 'Mesh_Crankshaft']) ok(!nombreGenerico(n), `«${n}» lo puso el autor: se respeta`);

// --- 4. Cómo se decide el nombre (con una red de juguete: solo importan las clases)
const juguete = {
  clases: ['piston', 'rotor', 'otro'], otro: 'otro', umbral_confianza: 0.6,
  info: { piston: { nombre: 'Pistón', descripcion: 'Sube y baja', familias: ['combustion'] },
    rotor: { nombre: 'Rotor', descripcion: 'Gira', familias: ['electrico'] }, otro: { familias: [] } },
};
const centros = [[3, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [5, 0, 0]];
const probs = [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.3, 0.65, 0.05]];
const nombres = nombrar(juguete, probs, centros, { familia: 'motor' });
ok(nombres[0]?.nombre === 'Pistón 2' && nombres[1]?.nombre === 'Pistón 1', 'las de una misma clase se numeran de izquierda a derecha');
ok(nombres[2] === null, 'si la red no está segura (50 % < 60 %), la pieza queda con su nombre');
ok(nombres[3] === null, '"otro" no se nombra');
ok(nombres[4]?.nombre === 'Rotor', 'una sola de su clase: sin número');
const combustion = nombrar(juguete, probs, centros, { familia: 'combustion' });
// Sin rotor posible, la 5.ª pasa a pistón (0,3 / 0,35 = 86 %) y también la 3.ª (0,5 / 0,7 = 71 %, antes 50 %)
ok(combustion[4]?.nombre === 'Pistón 4' && combustion[2]?.nombre === 'Pistón 2',
  'en un motor de combustión no hay rotor: la probabilidad se reparte entre las clases posibles');
ok(clasesPermitidas(juguete, 'electrico').join() === 'false,true,true', 'un motor eléctrico no tiene pistones');
const conMaximo = { ...juguete, info: { ...juguete.info, piston: { ...juguete.info.piston, maximo: 1 } } };
const unico = nombrar(conMaximo, probs, centros, { familia: 'motor' });
ok(unico[0]?.nombre === 'Pistón' && unico[1] === null,
  'si hay más de las que puede tener un motor, se queda con la de más confianza (0,9 > 0,8)');
const soloAlgunas = nombrar(juguete, probs, centros, { renombrables: [true, false, true, true, true] });
ok(soloAlgunas[1] === null && soloAlgunas[0]?.nombre === 'Pistón',
  'las piezas con nombre de autor no se tocan ni cuentan para numerar');

// --- 5. Igual que Python: rasgos y probabilidades del motor de prueba
const rutaRed = join(aqui, '..', 'ml', 'piezas_motor.json');
const rutaFixture = join(aqui, 'datos', 'piezas_fixture.json');
if (!existsSync(rutaRed) || !existsSync(rutaFixture)) {
  console.log('(sin red entrenada: python ml/piezas/entrenar_piezas.py exporta la red y el motor de prueba)');
} else {
  const red = leerRed(JSON.parse(readFileSync(rutaRed, 'utf8')));
  const f = JSON.parse(readFileSync(rutaFixture, 'utf8'));
  const muestras = f.muestras.map((pts) => Float64Array.from(pts.flat()));
  const { puntos, contexto } = rasgosDeMuestras(muestras, f.areas);
  let difPuntos = 0;
  let difContexto = 0;
  puntos.forEach((p, i) => p.forEach((v, k) => { difPuntos = Math.max(difPuntos, Math.abs(v - f.puntos[i].flat()[k])); }));
  contexto.forEach((c, i) => c.forEach((v, k) => { difContexto = Math.max(difContexto, Math.abs(v - f.contexto[i][k])); }));
  ok(difPuntos < 1e-3, `los puntos de cada pieza dan lo mismo que en Python (dif. máx. ${difPuntos.toExponential(1)})`);
  ok(difContexto < 1e-3, `el contexto de cada pieza da lo mismo que en Python (dif. máx. ${difContexto.toExponential(1)})`);
  // La red, con las mismas entradas que usó Python
  let difProb = 0;
  f.puntos.forEach((pts, i) => {
    const p = predecir(red, Float32Array.from(pts.flat()), f.contexto[i]);
    p.forEach((v, k) => { difProb = Math.max(difProb, Math.abs(v - f.probabilidades[i][k])); });
  });
  ok(difProb < 1e-4, `la red da las mismas probabilidades que PyTorch (dif. máx. ${difProb.toExponential(1)})`);
  // (El motor de prueba son solo 8 piezas de un motor de 26, para que el archivo sea chico: no sirve para medir
  // aciertos, porque la red mira cada pieza en el conjunto. Los aciertos están en ml/resultados/piezas_reporte.json.)
}

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
