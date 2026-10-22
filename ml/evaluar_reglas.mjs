// Clasifica cada fila del dataset con las reglas del navegador (frontend/js/gestures.js),
// fotograma a fotograma, para compararlas con el MLP sobre los mismos datos.
// Usa el código del visor tal cual: no hay una copia de las reglas en Python.
//
// Uso:  node ml/evaluar_reglas.mjs <directorio_csv> <salida.csv>
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { clasificarFotograma } from '../frontend/js/gestures.js';

const [directorio = 'ml/data', salida = 'ml/resultados/reglas_predicciones.csv'] = process.argv.slice(2);

const filas = ['archivo,fila,prediccion'];
for (const archivo of readdirSync(directorio).filter((a) => a.endsWith('.csv')).sort()) {
  const [cabecera, ...lineas] = readFileSync(join(directorio, archivo), 'utf8').trim().split(/\r?\n/);
  const columnas = cabecera.split(',');
  const indice = Object.fromEntries(columnas.map((c, i) => [c, i]));
  lineas.forEach((linea, fila) => {
    const v = linea.split(',');
    const puntos = Array.from({ length: 21 }, (_, i) => ({
      x: Number(v[indice[`x${i}`]]), y: Number(v[indice[`y${i}`]]), z: Number(v[indice[`z${i}`]]),
    }));
    filas.push(`${archivo},${fila},${clasificarFotograma(puntos, Number(v[indice.aspecto]))}`);
  });
}

mkdirSync(dirname(salida), { recursive: true });
writeFileSync(salida, filas.join('\n') + '\n');
console.log(`${filas.length - 1} filas clasificadas con reglas -> ${salida}`);
