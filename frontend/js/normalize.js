// Normalización de landmarks (sección 12.3 de la propuesta).
//
// Objetivo: que el mismo gesto produzca los mismos números sin importar dónde
// esté la mano en el cuadro ni qué tan cerca esté de la cámara.
//  - Traslación: el origen pasa a ser la muñeca (punto 0).
//  - Escala: se divide por la distancia muñeca → nudillo del dedo medio (punto 9),
//    que casi no cambia al mover los dedos y crece/achica con la distancia a la cámara.
//  - Rotación: se conserva. Las reglas usan distancias (que no dependen de la rotación)
//    y para el MLP la orientación puede ser información útil (p. ej. señalar).
//
// MediaPipe entrega x normalizada por el ancho de la imagen e y por el alto, así que
// primero se multiplica x (y z, que usa la misma escala que x) por la relación de aspecto
// para que las tres coordenadas queden en la misma unidad.

export const MUNECA = 0;
export const NUDILLO_MEDIO = 9;

export function normalizar(puntos, aspecto = 1) {
  const p = puntos.map((q) => ({ x: q.x * aspecto, y: q.y, z: q.z * aspecto }));
  const o = p[MUNECA];
  const r = p[NUDILLO_MEDIO];
  const escala = Math.hypot(r.x - o.x, r.y - o.y, r.z - o.z) || 1;
  return p.map((q) => ({ x: (q.x - o.x) / escala, y: (q.y - o.y) / escala, z: (q.z - o.z) / escala }));
}

// Vector plano de 63 valores (x0, y0, z0, ..., x20, y20, z20): entrada del MLP.
export function aVector(normalizados) {
  return normalizados.flatMap((q) => [q.x, q.y, q.z]);
}
