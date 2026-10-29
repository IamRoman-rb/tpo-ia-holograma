// Pedidos por voz (frontend/js/voz.js) con un reconocimiento de voz falso: no necesita micrófono.
// Uso (desde la raíz):  node frontend/tests/voz.test.mjs
import { DETECTOR, crearDetectorVoz, crearVoz, puedeReconocerLocal, reconocimientoDisponible } from '../js/voz.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

// Imita SpeechRecognition: guarda la última instancia para disparar sus eventos desde la prueba
let ultima = null;
class ReconocimientoFalso {
  constructor() { ultima = this; }
  start() { this.iniciado = true; }
  stop() { this.onend?.(); }
}
const resultado = (texto, isFinal) => ({ results: [Object.assign([{ transcript: texto }], { isFinal })] });

ok(reconocimientoDisponible({}) === null, 'sin la API (Firefox) no hay reconocimiento');
ok(reconocimientoDisponible({ webkitSpeechRecognition: ReconocimientoFalso }) === ReconocimientoFalso,
  'en Chrome se usa el prefijo webkit');
ok(crearVoz({}, null) === null, 'sin la API, crearVoz devuelve null y el botón se oculta');

const eventos = [];
const voz = crearVoz({
  alParcial: (t) => eventos.push(`parcial:${t}`),
  alFinal: (t) => eventos.push(`final:${t}`),
  alEstado: (e, d) => eventos.push(`estado:${e}${d ? `:${d}` : ''}`),
}, ReconocimientoFalso);

voz.escuchar();
ok(ultima.iniciado && ultima.lang === 'es-AR' && ultima.interimResults, 'escucha en español rioplatense, con resultados parciales');
ok(voz.escuchando && eventos.at(-1) === 'estado:escuchando', 'avisa que está escuchando');
ultima.onresult(resultado('mostrame la', false));
ultima.onresult(resultado('mostrame la válvula ', true));
ultima.onend();
ok(eventos.includes('parcial:mostrame la') && eventos.includes('final:mostrame la válvula'),
  `muestra lo parcial y entrega el pedido final (${eventos.join(' | ')})`);
ok(!voz.escuchando && eventos.at(-1) === 'final:mostrame la válvula', 'al terminar con un pedido no pisa el estado');

eventos.length = 0;
voz.escuchar();
ultima.onerror({ error: 'not-allowed' });
ultima.onend();
ok(eventos.includes('estado:error:No hay permiso para usar el micrófono.'), 'sin permiso de micrófono, un mensaje claro');

eventos.length = 0;
voz.escuchar();
ultima.onerror({ error: 'no-speech' });
ok(eventos.includes('estado:listo:No escuché nada.'), 'si no se habló, no es un error');

ok(await puedeReconocerLocal(ReconocimientoFalso) === false, 'si el navegador no informa reconocimiento local: false');
class ConLocal extends ReconocimientoFalso { static async available() { return 'available'; } }
ok(await puedeReconocerLocal(ConLocal) === true, 'si lo informa, se usa el reconocimiento en el equipo');
const local = crearVoz({ local: true }, class extends ConLocal { processLocally = false; });
local.escuchar();
ok(ultima.processLocally === true, 'con reconocimiento local, el audio no sale del equipo');

// Detector de voz con umbral adaptativo: ruido de fondo, una frase y silencio
let sem = 3;
const azar = () => ((sem = (sem * 1664525 + 1013904223) >>> 0) / 2 ** 32);
function simular(ruido, voz, desdeMs, hastaMs, totalMs = 4000) {
  const d = crearDetectorVoz();
  const eventos = [];
  for (let t = 0; t <= totalMs; t += 50) {
    const hablando = t >= desdeMs && t < hastaMs;
    const rms = ruido * (0.7 + 0.6 * azar()) + (hablando ? voz * (0.6 + 0.8 * azar()) : 0);
    const e = d.actualizar(rms, t);
    if (e !== 'silencio' && e !== 'voz') eventos.push([e, t]);
  }
  return { eventos, umbral: d.umbral };
}
for (const ruido of [0.002, 0.02]) {
  const { eventos, umbral } = simular(ruido, 0.08, 1500, 2600);
  const inicio = eventos.find(([e]) => e === 'inicio');
  const fin = eventos.find(([e]) => e === 'fin');
  ok(inicio && inicio[1] >= 1500 && inicio[1] <= 1500 + DETECTOR.inicioMs + 100,
    `ruido ${ruido}: detecta el inicio de la frase (a los ${inicio?.[1]} ms; umbral ${umbral.toFixed(3)})`);
  ok(fin && fin[1] >= 2600 + DETECTOR.silencioMs - 100 && fin[1] <= 2600 + DETECTOR.silencioMs + 200,
    `ruido ${ruido}: cierra la frase después del silencio (a los ${fin?.[1]} ms)`);
}
const golpe = simular(0.003, 0.1, 1500, 1560);
ok(!golpe.eventos.some(([e]) => e === 'inicio'), 'un golpe corto no cuenta como una frase');
const larga = simular(0.003, 0.08, 500, 20000, 20000);
ok(larga.eventos.some(([e, t]) => e === 'fin' && t <= 500 + DETECTOR.maximoMs + 100), 'una frase interminable se corta sola');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
