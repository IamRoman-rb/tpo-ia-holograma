// Lógica del desmontaje por piezas (sin Three.js, para poder probarla en Node).
//
// Cada pieza llega del .glb con sus metadatos (backend/geometria.py los escribe en "extras"):
//   { id, nombre, descripcion, orden, direccion, centro, volumen }
// orden: 1 = se saca primero · sin orden = pieza fija (por ejemplo, el cuerpo).
// direccion: hacia dónde se desliza al salir.

const sumar = (a, b) => a.map((v, i) => v + b[i]);
const restar = (a, b) => a.map((v, i) => v - b[i]);
const punto = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const largo = (a) => Math.hypot(...a);
const unitario = (a) => {
  const l = largo(a);
  return l > 1e-6 ? a.map((v) => v / l) : null;
};

/**
 * Completa orden, dirección y distancia de desmontaje.
 * centroModelo, tamano: centro y diagonal de la caja del modelo completo.
 * Devuelve las piezas ordenadas: primero las que se desmontan (por orden), al final las fijas.
 */
export function prepararPiezas(datos, centroModelo, tamano) {
  let piezas = datos.map((p) => ({ ...p }));
  const conOrden = piezas.filter((p) => Number.isFinite(p.orden));

  if (conOrden.length === 0 && piezas.length > 1) {
    // Sin orden (por ejemplo, un modelo generado sin metadatos): queda fija la pieza más
    // grande y las demás salen desde la más externa hacia adentro.
    const mayor = piezas.reduce((a, b) => ((b.volumen ?? 0) > (a.volumen ?? 0) ? b : a));
    piezas
      .filter((p) => p !== mayor)
      .sort((a, b) => largo(restar(b.centro, centroModelo)) - largo(restar(a.centro, centroModelo)))
      .forEach((p, i) => { p.orden = i + 1; });
  }

  for (const p of piezas) {
    p.fija = !Number.isFinite(p.orden);
    const desdeCentro = restar(p.centro, centroModelo);
    p.dir = unitario(p.direccion ?? []) ?? unitario(desdeCentro) ?? [0, 1, 0];
    // Las piezas más alejadas en su dirección se alejan más: así no quedan superpuestas
    p.dist = tamano * 0.35 + Math.max(0, punto(desdeCentro, p.dir));
  }

  // Entre piezas que salen en la misma dirección, la que se saca antes queda más afuera
  // (como en un despiece técnico: los bulones por fuera de la brida que sujetan)
  const desmontables = piezas.filter((p) => !p.fija).sort((a, b) => a.orden - b.orden);
  for (const p of desmontables) {
    const despues = desmontables.filter((q) => q.orden > p.orden && punto(q.dir, p.dir) > 0.9).length;
    p.dist += despues * tamano * 0.15;
  }

  piezas = piezas.sort((a, b) => (a.fija - b.fija) || ((a.orden ?? 0) - (b.orden ?? 0)));
  return piezas;
}

/** Índice de la próxima pieza para desmontar, o -1 si ya están todas afuera. */
export function siguienteParaDesmontar(piezas, objetivos) {
  return piezas.findIndex((p, i) => !p.fija && objetivos[i] === 0);
}

/** Índice de la última pieza desmontada (la que se vuelve a montar primero), o -1. */
export function ultimaDesmontada(piezas, objetivos) {
  for (let i = piezas.length - 1; i >= 0; i--) if (!piezas[i].fija && objetivos[i] > 0) return i;
  return -1;
}

/** Posición desplazada de una pieza: base + dirección · distancia · fracción (0 = montada, 1 = afuera). */
export function desplazamiento(pieza, fraccion) {
  return sumar([0, 0, 0], pieza.dir.map((v) => v * pieza.dist * fraccion));
}
