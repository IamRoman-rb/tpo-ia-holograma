// Movimiento de las piezas cuando el objeto está "en marcha" (sin Three.js, para poder probarlo en Node).
//
// Los modelos generados (backend/motores/, el exoesqueleto) traen en los extras de cada pieza cómo se mueve, y en
// los de la escena la velocidad del ciclo (rad/s). Todo depende de un único ángulo θ: el del cigüeñal en un motor,
// el del paso en el exoesqueleto. Las coordenadas de las mallas ya están en el sistema del modelo, así que cada
// pieza se mueve con una rotación q y una traslación t:  punto movido = q·punto + t.
//
//   giro:    gira relacion·θ alrededor del eje que pasa por centro (cigüeñal, levas, rotor, volante).
//   piston:  se desliza por su cilindro según la manivela y la biela. El muñón está en
//            r·(0, cos(fase + θ), sin(fase + θ)) y el perno del pistón, a lo largo del cilindro u, en
//            s = muñón·u + √(L² − (muñón·w)²), con w perpendicular a u en el plano del giro. Se mueve s − s0.
//   biela:   va del muñón al perno: se lleva el extremo de abajo al muñón nuevo y se la gira para que apunte al perno.
//   oscilar: gira intensidad·(base + amplitud·sin(relacion·θ + fase)) alrededor del eje (articulaciones). La
//            intensidad sube de 0 a 1 al encender y baja al apagar: arranca y frena suave, y quieta queda como se
//            dibujó. Con padre, además acompaña el movimiento de esa pieza: el antebrazo sigue al brazo.

/** θ en el que todas las piezas vuelven a como se dibujaron (las levas van a media vuelta: hacen falta dos). */
export const PERIODO = 4 * Math.PI;

const IDENTIDAD = Object.freeze({ q: [0, 0, 0, 1], t: [0, 0, 0] });

const unitario = (v) => {
  const l = Math.hypot(...v);
  return l > 1e-9 ? v.map((x) => x / l) : [1, 0, 0];
};

export function cuaternionEje(eje, angulo) {
  const [x, y, z] = unitario(eje);
  const s = Math.sin(angulo / 2);
  return [x * s, y * s, z * s, Math.cos(angulo / 2)];
}

export function multiplicar(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function rotar(q, v) {
  const [x, y, z, w] = q;
  // v' = v + 2w(q×v) + 2 q×(q×v)
  const cx = y * v[2] - z * v[1];
  const cy = z * v[0] - x * v[2];
  const cz = x * v[1] - y * v[0];
  return [
    v[0] + 2 * (w * cx + y * cz - z * cy),
    v[1] + 2 * (w * cy + z * cx - x * cz),
    v[2] + 2 * (w * cz + x * cy - y * cx),
  ];
}

/** Rotación alrededor de un eje que pasa por centro, como { q, t }. */
function girarEn(eje, centro, angulo) {
  const q = cuaternionEje(eje, angulo);
  const c = centro ?? [0, 0, 0];
  const rc = rotar(q, c);
  return { q, t: [c[0] - rc[0], c[1] - rc[1], c[2] - rc[2]] };
}

/** Muñón y perno de un cilindro para el ángulo θ (en el sistema del modelo). */
export function manivela(m, theta) {
  const u = unitario(m.u);
  const w = [0, -u[2], u[1]];
  const fi = (m.fase ?? 0) + theta;
  const munon = [m.x ?? 0, m.r * Math.cos(fi), m.r * Math.sin(fi)];
  const a = munon[1] * u[1] + munon[2] * u[2];
  const b = munon[1] * w[1] + munon[2] * w[2];
  const s = a + Math.sqrt(Math.max(m.L * m.L - b * b, 1e-9));
  const perno = [m.x ?? 0, u[1] * s, u[2] * s];
  return { munon, perno, s, u };
}

/** Movimiento propio de una pieza (sin el de su padre). intensidad: solo para las articulaciones. */
export function movimientoPropio(m, theta, intensidad = 1) {
  if (!m) return IDENTIDAD;
  switch (m.tipo) {
    case 'giro':
      return girarEn(m.eje ?? [1, 0, 0], m.centro, (m.relacion ?? 1) * theta);
    case 'piston': {
      const { s, u } = manivela(m, theta);
      const d = s - m.s0;
      return { q: [0, 0, 0, 1], t: [u[0] * d, u[1] * d, u[2] * d] };
    }
    case 'biela': {
      // Como se dibujó: del muñón en θ = 0 al perno en s0. Ahora: del muñón nuevo al perno nuevo.
      const u = unitario(m.u);
      const m0 = [m.x ?? 0, m.r * Math.cos(m.fase ?? 0), m.r * Math.sin(m.fase ?? 0)];
      const p0 = [m.x ?? 0, u[1] * m.s0, u[2] * m.s0];
      const { munon, perno } = manivela(m, theta);
      const antes = Math.atan2(p0[2] - m0[2], p0[1] - m0[1]);
      const ahora = Math.atan2(perno[2] - munon[2], perno[1] - munon[1]);
      const q = cuaternionEje([1, 0, 0], ahora - antes);
      const r0 = rotar(q, m0);
      return { q, t: [munon[0] - r0[0], munon[1] - r0[1], munon[2] - r0[2]] };
    }
    case 'oscilar': {
      const k = m.relacion ?? 1;
      const angulo = intensidad * ((m.base ?? 0) + (m.amplitud ?? 0) * Math.sin(k * theta + (m.fase ?? 0)));
      return girarEn(m.eje ?? [1, 0, 0], m.centro, angulo);
    }
    default:
      return IDENTIDAD;
  }
}

/** Primero el movimiento propio y después el del padre: punto → padre(propio(punto)). */
export function componer(padre, propio) {
  const t = rotar(padre.q, propio.t);
  return { q: multiplicar(padre.q, propio.q), t: [t[0] + padre.t[0], t[1] + padre.t[1], t[2] + padre.t[2]] };
}

/**
 * Movimiento de cada pieza para el ángulo θ. piezas: [{ id, movimiento }]. Devuelve un Map id -> { q, t }
 * (solo de las que se mueven). Un padre que no existe o que forma un ciclo se ignora.
 */
export function movimientos(piezas, theta, intensidad = 1) {
  const porId = new Map(piezas.map((p) => [p.id, p.movimiento]));
  const listo = new Map();
  const calcular = (id, visitados = new Set()) => {
    if (listo.has(id)) return listo.get(id);
    const m = porId.get(id);
    let r = movimientoPropio(m, theta, intensidad);
    if (m?.padre && porId.get(m.padre) && !visitados.has(m.padre)) {
      visitados.add(id);
      r = componer(calcular(m.padre, visitados), r);
    }
    listo.set(id, r);
    return r;
  };
  for (const p of piezas) if (p.movimiento) calcular(p.id);
  return listo;
}

export const RAMPA_S = 0.6;   // lo que tardan las articulaciones en llegar a todo su movimiento (o en aquietarse)

/**
 * Reloj de la marcha: avanzar(dt) devuelve el θ actual. Al apagar, sigue girando hasta completar el ciclo y se
 * detiene donde las piezas están como se dibujaron (así el despiece y las etiquetas vuelven a coincidir).
 * intensidad: de 0 a 1, para las articulaciones (sube al encender y baja al apagar, en RAMPA_S segundos).
 */
export function crearMarcha(velocidad) {
  let theta = 0;
  let encendida = false;
  let parada = 0;
  let intensidad = 0;
  return {
    get theta() { return theta; },
    get intensidad() { return intensidad; },
    get encendida() { return encendida; },
    get moviendose() { return encendida || theta !== 0 || intensidad > 0; },
    encender() { encendida = true; },
    apagar() {
      if (!encendida) return;
      encendida = false;
      parada = Math.ceil(theta / PERIODO) * PERIODO;
      if (parada - theta < 1e-6) theta = 0;
    },
    avanzar(dt) {
      if (!velocidad) return theta;
      intensidad = Math.min(1, Math.max(0, intensidad + (encendida ? dt : -dt) / RAMPA_S));
      if (encendida) {
        theta = (theta + velocidad * dt) % (PERIODO * 1000);
      } else if (theta !== 0) {
        theta += velocidad * dt;
        if (theta >= parada) theta = 0;
      }
      return theta;
    },
  };
}
