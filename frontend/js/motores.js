// Los motores de IA del sistema y las respuestas a los comandos para manejarlos ("usá el árbol para los
// gestos", "¿qué modelo respondió?", "¿qué tan seguro estás?", "¿por qué?"). Sin DOM: se prueba en Node.

// Cómo se llama cada motor de los pedidos (el campo "motor" que devuelve el backend), corto y para decirlo
export const MOTORES_PEDIDOS = {
  llm: { corto: 'Claude', largo: 'Claude, el LLM de Anthropic' },
  claude: { corto: 'Claude', largo: 'Claude, el LLM de Anthropic' },
  consenso: { corto: 'consenso', largo: 'el consenso de todos los modelos' },
  llm_propio: { corto: 'LLM propio', largo: 'el LLM propio (Qwen ajustado)' },
  transformer: { corto: 'transformer', largo: 'la red transformer (RoBERTa ajustada)' },
  modelo: { corto: 'modelo propio', largo: 'el intérprete con embeddings' },
  bayes: { corto: 'Naive Bayes', largo: 'el modelo estadístico Naive Bayes' },
  reglas: { corto: 'reglas', largo: 'las reglas' },
  automatico: { corto: 'automático', largo: 'la cadena automática (Claude, si hay, y si no el consenso)' },
};

// Clasificadores de gestos: el nombre del backend (usar_motor, ámbito gestos) → el del visor
export const CLASIFICADOR_DE_MOTOR = { arbol: 'arbol', red: 'mlp', reglas: 'reglas', automatico: 'mlp' };
export const MOTORES_GESTOS = { reglas: 'las reglas', mlp: 'la red neuronal', arbol: 'el árbol de decisión' };

export const nombreMotor = (motor, largo = false) =>
  MOTORES_PEDIDOS[motor]?.[largo ? 'largo' : 'corto'] ?? motor ?? 'reglas';

const NOMBRES_ACCION = {
  mostrar: 'mostrar', rotar: 'rotar', escalar: 'escalar', vista: 'cambiar la vista', reiniciar: 'reiniciar',
  desmontar: 'desmontar una pieza', explotar: 'mostrar el despiece', armar: 'armar todo', aislar: 'analizar la pieza',
  conjunto: 'volver al conjunto', usar_motor: 'cambiar de motor', consultar: 'responder una consulta',
  desconocido: 'no reconocido',
};
export const nombreAccion = (a) => NOMBRES_ACCION[a] ?? a;

const porcentaje = (v) => `${Math.round((v ?? 0) * 100)} %`;

/** Explicación de un pedido a partir de su traza (lo que el backend contó de cómo decidió). */
export function explicarPedido(i) {
  const t = i.traza ?? {};
  const partes = [`Lo interpretó ${nombreMotor(i.motor, true)}`];
  if (t.votos?.length) {
    // consenso: qué votó cada modelo
    const votos = t.votos.map((v) => `${nombreMotor(v.motor)} ${nombreAccion(v.accion)} (${porcentaje(v.confianza)})`);
    partes.push(`votaron ${votos.join(', ')}`);
    if (t.decision) partes.push(t.decision);
  } else if (t.probabilidades) {
    const [primera, segunda] = Object.entries(t.probabilidades);
    partes.push(`le dio ${porcentaje(primera?.[1])} a ${nombreAccion(primera?.[0])}`
      + (segunda ? ` y ${porcentaje(segunda[1])} a ${nombreAccion(segunda[0])}` : ''));
  }
  if (t.rasgos_decisivos?.length) partes.push(`las palabras que más pesaron: ${t.rasgos_decisivos.join(', ').replaceAll('_', ' ')}`);
  if (t.crudo_llm_propio) partes.push(`el LLM escribió «${t.crudo_llm_propio}»`);
  if (i.objeto) partes.push(`el objeto es «${i.objeto}»`);
  return `${partes.join('; ')}.`;
}

/**
 * Respuesta a un comando de los modelos. contexto: { ultimo: el último pedido atendido (intención con motor y
 * traza), gestos: { metodo, gesto, confianza, camino } del último cuadro, aplicado: lo que hizo usar_motor }.
 */
export function respuestaComando(intencion, { ultimo = null, gestos = null, aplicado = null } = {}) {
  const p = intencion.parametros ?? {};
  if (intencion.accion === 'usar_motor') {
    if (p.ambito === 'gestos') return `Listo: los gestos los reconoce ${MOTORES_GESTOS[aplicado ?? CLASIFICADOR_DE_MOTOR[p.motor]] ?? p.motor}.`;
    return p.motor === 'automatico' ? `Listo: vuelvo a ${nombreMotor('automatico', true)}.`
      : `Listo: los pedidos los interpreta ${nombreMotor(p.motor, true)}.`;
  }
  const sobreGestos = gestos && /gest|mano/.test(intencion.frase ?? '');
  if (!ultimo && !sobreGestos) return 'Todavía no respondí ningún pedido.';
  if (sobreGestos) {
    const g = `${gestos.gesto ?? 'ninguno'} con ${MOTORES_GESTOS[gestos.metodo] ?? gestos.metodo}`;
    if (p.pregunta === 'confianza') return `El gesto ${g}: confianza ${porcentaje(gestos.confianza)}.`;
    if (p.pregunta === 'explicar' && gestos.camino) return `El árbol siguió este camino: ${gestos.camino}.`;
    return `Los gestos los reconoce ${MOTORES_GESTOS[gestos.metodo] ?? gestos.metodo}; el último fue ${g}.`;
  }
  if (p.pregunta === 'que_motor') return `El último pedido lo interpretó ${nombreMotor(ultimo.motor, true)}.`;
  if (p.pregunta === 'confianza') {
    return `Estaba ${porcentaje(ultimo.confianza)} seguro de ${nombreAccion(ultimo.accion)}`
      + `${ultimo.objeto ? ` «${ultimo.objeto}»` : ''} (${nombreMotor(ultimo.motor)}).`;
  }
  return explicarPedido(ultimo);
}
