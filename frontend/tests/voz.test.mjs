// Pedidos por voz (frontend/js/voz.js) con un reconocimiento de voz falso: no necesita micrófono.
// Uso (desde la raíz):  node frontend/tests/voz.test.mjs
import { crearVoz, puedeReconocerLocal, reconocimientoDisponible } from '../js/voz.js';

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
const resultados = (...partes) => ({
  results: partes.map(([texto, isFinal]) => Object.assign([{ transcript: texto }], { isFinal })),
});

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
ultima.onresult(resultados(['mostrame la', true], ['válvula', false]));
ok(!eventos.some((e) => e.startsWith('final:')), 'no envía una transcripción parcial aunque haya resultados finales previos');
ultima.onresult(resultados(['mostrame la', true], ['válvula', true]));
ultima.onend();
ok(eventos.includes('parcial:mostrame la') && eventos.includes('final:mostrame la válvula'),
  `muestra lo parcial y entrega el pedido final (${eventos.join(' | ')})`);
ok(!voz.escuchando && eventos.at(-1) === 'final:mostrame la válvula', 'al terminar con un pedido no pisa el estado');

eventos.length = 0;
voz.escuchar();
ultima.onerror({ error: 'not-allowed' });
ultima.onend();
ok(eventos.at(-1) === 'estado:error:No hay permiso para usar el micrófono.' && !voz.escuchando,
  'el error de permiso se conserva al terminar el reconocimiento');

eventos.length = 0;
voz.escuchar();
ultima.onerror({ error: 'no-speech' });
ok(eventos.includes('estado:listo:No escuché nada.'), 'si no se habló, no es un error');

class InicioFallido extends ReconocimientoFalso { start() { throw new DOMException('denegado', 'NotAllowedError'); } }
const fallido = crearVoz({ alEstado: (e, d) => eventos.push(`estado:${e}:${d}`) }, InicioFallido);
fallido.escuchar();
ok(!fallido.escuchando && eventos.at(-1) === 'estado:error:No hay permiso para usar el micrófono.',
  'si el navegador rechaza el inicio, informa el permiso y restablece el estado');

ok(await puedeReconocerLocal(ReconocimientoFalso) === false, 'si el navegador no informa reconocimiento local: false');
class ConLocal extends ReconocimientoFalso { static async available() { return 'available'; } }
ok(await puedeReconocerLocal(ConLocal) === true, 'si lo informa, se usa el reconocimiento en el equipo');
const local = crearVoz({ local: true }, class extends ConLocal { processLocally = false; });
local.escuchar();
ok(ultima.processLocally === true, 'con reconocimiento local, el audio no sale del equipo');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
