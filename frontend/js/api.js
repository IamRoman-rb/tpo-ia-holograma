// Comunicación con el backend FastAPI.
const API = 'http://localhost:8000';

async function post(ruta, cuerpo) {
  const r = await fetch(`${API}${ruta}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cuerpo),
  });
  if (!r.ok) throw new Error(`${ruta} respondió ${r.status}`);
  return r.json();
}

export const interpretarPedido = (frase) => post('/intent', { frase });
export const pedirObjeto = (frase) => post('/object', { frase });
export const salud = () => fetch(`${API}/health`).then((r) => r.json());
