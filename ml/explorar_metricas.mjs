// Distribución de las métricas de las reglas por clase, para calibrar umbrales con datos y no a ojo.
// Usar SOLO la partición de entrenamiento: la de prueba se reserva para medir.
//
// Uso:  node ml/explorar_metricas.mjs ml/data/externo/entrenamiento/hagrid_entrenamiento.csv
import { readFileSync } from 'node:fs';
import { medir } from '../frontend/js/gestures.js';
import { normalizar } from '../frontend/js/normalize.js';

const [archivo] = process.argv.slice(2);
const [cabecera, ...lineas] = readFileSync(archivo, 'utf8').trim().split(/\r?\n/);
const indice = Object.fromEntries(cabecera.split(',').map((c, i) => [c, i]));

const porClase = {};
for (const linea of lineas) {
  const v = linea.split(',');
  const puntos = Array.from({ length: 21 }, (_, i) => ({
    x: Number(v[indice[`x${i}`]]), y: Number(v[indice[`y${i}`]]), z: Number(v[indice[`z${i}`]]),
  }));
  const m = medir(normalizar(puntos, Number(v[indice.aspecto])));
  (porClase[v[indice.clase]] ??= []).push(m);
}

const cuantil = (xs, q) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
const metricas = Object.keys(Object.values(porClase)[0][0]);
for (const metrica of metricas) {
  console.log(`\n${metrica}          p05    p25    p50    p75    p95`);
  for (const [clase, ms] of Object.entries(porClase)) {
    const xs = ms.map((m) => m[metrica]);
    console.log(`  ${clase.padEnd(14)} ${[0.05, 0.25, 0.5, 0.75, 0.95].map((q) => cuantil(xs, q).toFixed(2).padStart(6)).join(' ')}`);
  }
}
