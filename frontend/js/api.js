// Comunicación con el backend FastAPI.
const API = 'http://localhost:8000';
const ESPERA_MS = 20000; // el LLM puede tardar unos segundos

export class ErrorBackend extends Error {
  constructor(mensaje, estado) {
    super(mensaje);
    this.estado = estado; // undefined = el backend no respondió
  }
}

async function post(ruta, cuerpo) {
  let r;
  try {
    r = await fetch(`${API}${ruta}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cuerpo),
      signal: AbortSignal.timeout(ESPERA_MS),
    });
  } catch (err) {
    throw new ErrorBackend(`backend no disponible (${err.name})`);
  }
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new ErrorBackend(datos.detail ?? `${ruta} respondió ${r.status}`, r.status);
  return datos;
}

export const interpretarPedido = (frase) => post('/intent', { frase });
export const pedirObjeto = (objeto) => post('/object', { frase: objeto });
export const salud = () => fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) }).then((r) => r.json());
