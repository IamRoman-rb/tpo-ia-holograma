// Catálogo precargado (frontend/models/catalogo.json) y búsqueda local por palabras.
// Es el plan B cuando el backend no responde. En la fase 3 la búsqueda principal
// pasa al backend con embeddings (backend/retrieval.py).

const RELLENO = new Set([
  'mostrame', 'mostra', 'mostrar', 'quiero', 'ver', 'dame', 'traeme', 'trae', 'busca', 'buscame',
  'un', 'una', 'unos', 'unas', 'el', 'la', 'los', 'las', 'de', 'del', 'con', 'por', 'favor', 'me', 'a', 'y',
]);

export const UMBRAL_LOCAL = 0.5;

function palabras(texto) {
  return texto
    .toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')   // sin tildes
    .replace(/[^a-z0-9ñ\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w && !RELLENO.has(w))
    .map((w) => (w.length > 4 && w.endsWith('s') ? w.slice(0, -1) : w)); // plural simple
}

export async function cargarCatalogo() {
  const r = await fetch('models/catalogo.json', { cache: 'no-store' }); // puede cambiar al generar modelos
  if (!r.ok) throw new Error(`catalogo.json respondió ${r.status}`);
  return r.json();
}

// Devuelve { item, similitud } o null. similitud = fracción de las palabras del
// nombre o sinónimo que aparecen en el pedido (se toma la mejor variante).
export function buscarLocal(frase, items) {
  const pedido = new Set(palabras(frase));
  if (pedido.size === 0) return null;
  let mejor = null;
  for (const item of items) {
    for (const variante of [item.nombre, ...item.sinonimos]) {
      const p = palabras(variante);
      if (p.length === 0) continue;
      const sim = p.filter((w) => pedido.has(w)).length / p.length;
      if (!mejor || sim > mejor.similitud) mejor = { item, similitud: sim };
    }
  }
  return mejor && mejor.similitud >= UMBRAL_LOCAL ? mejor : null;
}
