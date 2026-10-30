// Atribución de los modelos de Objaverse (CC BY pide nombrar la obra, el autor y la licencia).
// Uso (desde la raíz):  node frontend/tests/credito.test.mjs
import { creditoModelo } from '../js/ui.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

const credito = creditoModelo({ titulo: 'Engine V8', autor: 'Fulana', licencia: 'CC BY 4.0', url: 'https://x' });
ok(credito === '"Engine V8" de Fulana · CC BY 4.0', `título, autor y licencia: ${credito}`);
ok(creditoModelo({ titulo: 'Gear', autor: '', licencia: 'CC0 1.0' }).includes('autor desconocido'),
  'sin autor no queda un hueco');

if (fallas) process.exit(1);
