// Definición del dataset de gestos: clases y formato CSV (sección 12.2).
// Lo usan la herramienta de captura y los tests; ml/preparar.py lee este mismo formato.

export const CLASES = [
  { id: 'pellizco', nombre: 'Pellizco', descripcion: 'Pulgar e índice juntos. En el visor: tomar y mover el objeto.' },
  { id: 'palma_abierta', nombre: 'Palma abierta', descripcion: 'Mano abierta frente a la cámara. En el visor: rotar.' },
  { id: 'puño', nombre: 'Puño', descripcion: 'Mano cerrada. En el visor: reiniciar la vista.' },
  { id: 'señalar', nombre: 'Señalar', descripcion: 'Índice extendido y el resto de los dedos cerrados. En el visor: seleccionar o marcar.' },
  { id: 'ninguno', nombre: 'Ninguno', descripcion: 'Mano relajada, a medio cerrar, o movimientos al azar. Es fundamental: sin esta clase el modelo ve un gesto en cualquier cosa.' },
];

export const PERSONA_SIMULADA = 'simulacion'; // ml/preparar.py descarta estas filas por defecto

export function aCsv(muestras) {
  const columnas = ['clase', 'persona', 'mano', 'aspecto'];
  for (let i = 0; i < 21; i++) columnas.push(`x${i}`, `y${i}`, `z${i}`);
  const filas = muestras.map((m) => [
    m.clase, m.persona, m.mano, m.aspecto.toFixed(4),
    ...m.puntos.flatMap((p) => [p.x.toFixed(5), p.y.toFixed(5), p.z.toFixed(5)]),
  ].join(','));
  return [columnas.join(','), ...filas].join('\n') + '\n';
}
