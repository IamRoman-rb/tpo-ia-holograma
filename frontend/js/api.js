// Comunicación con el backend FastAPI.
const API = 'http://localhost:8000';
const ESPERA_MS = 20000;              // el LLM puede tardar unos segundos
const ESPERA_GENERACION_MS = 330000;  // generar un modelo 3D puede llevar minutos

export class ErrorBackend extends Error {
  constructor(mensaje, estado) {
    super(mensaje);
    this.estado = estado; // undefined = el backend no respondió
  }
}

async function post(ruta, cuerpo, espera = ESPERA_MS) {
  let r;
  try {
    r = await fetch(`${API}${ruta}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cuerpo),
      signal: AbortSignal.timeout(espera),
    });
  } catch (err) {
    throw new ErrorBackend(`backend no disponible (${err.name})`);
  }
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new ErrorBackend(datos.detail ?? `${ruta} respondió ${r.status}`, r.status);
  return datos;
}

// motor: el elegido por voz ("usá el LLM propio"); sin él, la cadena por defecto del backend
export const interpretarPedido = (frase, motor = null) => post('/intent', motor ? { frase, motor } : { frase });

/** Audio de una frase → texto, con Whisper en el backend (backend/voz.py). activacion: modo manos libres. */
export async function transcribir(audio, { activacion = false } = {}) {
  let r;
  try {
    r = await fetch(`${API}/voz${activacion ? '?activacion=true' : ''}`, {
      method: 'POST', headers: { 'Content-Type': audio.type || 'audio/webm' }, body: audio,
      signal: AbortSignal.timeout(60000),   // Whisper large-v3 en el procesador tarda unos segundos
    });
  } catch (err) {
    throw new ErrorBackend(`backend no disponible (${err.name})`);
  }
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new ErrorBackend(datos.detail ?? `/voz respondió ${r.status}`, r.status);
  return datos;
}
export const pedirObjeto = (objeto) => post('/object', { frase: objeto }, ESPERA_GENERACION_MS);
export const salud = () => fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) }).then((r) => r.json());
