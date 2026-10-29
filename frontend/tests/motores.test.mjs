// Respuestas a los comandos de los modelos (frontend/js/motores.js): "usá el árbol para los gestos",
// "¿qué modelo respondió?", "¿qué tan seguro estás?", "¿por qué?". Uso:  node frontend/tests/motores.test.mjs
import { CLASIFICADOR_DE_MOTOR, explicarPedido, nombreMotor, respuestaComando } from '../js/motores.js';

let fallas = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'OK   ' : 'FALLA'} ${msg}`); if (!cond) fallas++; };

const usar = (motor, ambito) => ({ accion: 'usar_motor', parametros: { motor, ambito } });
const consultar = (pregunta, frase = '') => ({ accion: 'consultar', parametros: { pregunta }, frase });

ok(respuestaComando(usar('arbol', 'gestos'), { aplicado: 'arbol' }) === 'Listo: los gestos los reconoce el árbol de decisión.',
  'usar el árbol para los gestos');
ok(respuestaComando(usar('red', 'gestos')).includes('la red neuronal'), 'la red de gestos se llama "red" en el backend y "mlp" en el visor');
ok(CLASIFICADOR_DE_MOTOR.automatico === 'mlp', 'automático, para los gestos, es la red neuronal');
ok(respuestaComando(usar('bayes', 'pedidos')).includes('Naive Bayes'), 'usar Naive Bayes para los pedidos');
ok(respuestaComando(usar('automatico', 'pedidos')).startsWith('Listo: vuelvo a'), 'volver a la cadena automática');

ok(respuestaComando(consultar('que_motor')) === 'Todavía no respondí ningún pedido.', 'sin pedidos todavía');
const ultimo = {
  accion: 'mostrar', objeto: 'engranaje', confianza: 0.93, motor: 'consenso',
  traza: { votos: [{ motor: 'llm_propio', accion: 'mostrar', confianza: 0.98 }, { motor: 'bayes', accion: 'mostrar', confianza: 0.81 }],
    decision: 'ejecutar: el costo esperado de equivocarse (0,07) es menor que el de preguntar (0,2)' },
};
ok(respuestaComando(consultar('que_motor'), { ultimo }).includes('el consenso de todos los modelos'), '¿qué modelo respondió?');
ok(respuestaComando(consultar('confianza'), { ultimo }) === 'Estaba 93 % seguro de mostrar «engranaje» (consenso).',
  '¿qué tan seguro estás?');
const explicacion = respuestaComando(consultar('explicar'), { ultimo });
ok(explicacion.includes('LLM propio mostrar (98 %)') && explicacion.includes('costo esperado'), `¿por qué?: ${explicacion}`);

const gestos = { metodo: 'arbol', gesto: 'señalar', confianza: 0.83, camino: 'índice estirado > 1,20 → señalar' };
ok(respuestaComando(consultar('explicar', 'por qué ese gesto'), { ultimo, gestos }).includes('índice estirado'),
  'una pregunta sobre el gesto explica el camino del árbol');
ok(explicarPedido({ motor: 'bayes', traza: { probabilidades: { rotar: 0.9, vista: 0.05 }, rasgos_decisivos: ['giralo', '<i>_giralo'] } })
  .includes('le dio 90 % a rotar'), 'Naive Bayes explica con sus probabilidades y las palabras que más pesaron');
ok(nombreMotor('llm') === 'Claude' && nombreMotor(undefined) === 'reglas', 'nombres cortos de los motores');

console.log(fallas ? `\n${fallas} falla(s)` : '\nTodo OK');
process.exit(fallas ? 1 : 0);
