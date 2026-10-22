// Registro de latencias en el navegador (sección 15).
//   pedido_catalogo:  desde que se envía el pedido hasta que el modelo se dibuja (meta < 2 s)
//   pedido_generado:  lo mismo cuando el modelo se generó con IA (se reporta)
//   fotograma:        procesamiento de un fotograma de la cámara: detección + gesto + dibujo (meta < 100 ms)
//   fotograma_simulado: lo mismo con manos simuladas (sin el tiempo de MediaPipe)
// Se ven en el panel de depuración (tecla D). Desde la consola: metricas.csv()

const MAXIMO = 500;
const registros = {};

export function registrar(tipo, ms) {
  const lista = (registros[tipo] ??= []);
  lista.push(ms);
  if (lista.length > MAXIMO) lista.shift();
}

function percentil(ordenados, p) {
  return ordenados[Math.min(ordenados.length - 1, Math.floor(p * ordenados.length))];
}

export function resumen(tipo) {
  const lista = registros[tipo];
  if (!lista?.length) return null;
  const o = [...lista].sort((a, b) => a - b);
  return { n: o.length, mediana: percentil(o, 0.5), p95: percentil(o, 0.95), maximo: o.at(-1) };
}

export function lineasResumen() {
  return Object.keys(registros).map((tipo) => {
    const r = resumen(tipo);
    return `${tipo}: ${r.mediana.toFixed(1)} ms (p95 ${r.p95.toFixed(1)}, n=${r.n})`;
  });
}

export function csv() {
  const filas = ['tipo,ms'];
  for (const [tipo, lista] of Object.entries(registros)) for (const ms of lista) filas.push(`${tipo},${ms.toFixed(2)}`);
  return filas.join('\n');
}

// Espera a que el navegador dibuje el próximo cuadro (el modelo ya está en pantalla)
export const proximoCuadro = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

globalThis.metricas = { resumen, csv, registros };
