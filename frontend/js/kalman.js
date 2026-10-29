// Filtro de Kalman para los puntos de la mano (método estadístico de estimación).
//
// MediaPipe da cada punto con un temblor de unos milésimos de la imagen: en un pellizco eso mueve el objeto
// aunque la mano esté quieta. El filtro estima la posición real como una variable que se mueve con velocidad
// casi constante (modelo de velocidad constante con aceleración aleatoria) y que se mide con ruido:
//   predicción:   x ← x + v·dt                   P ← F P Fᵀ + Q(dt)
//   corrección:   K = P Hᵀ / (H P Hᵀ + R)         x ← x + K (medida − x)        P ← (I − K H) P
// Cada coordenada (21 puntos × x, y, z) tiene su filtro de dos estados (posición y velocidad).
// R es la varianza del ruido de medición y q la intensidad de la aceleración: con q chico el filtro suaviza
// más pero se atrasa en los movimientos rápidos; por eso q se adapta (ver crearMano). q y la adaptación salen de
// frontend/tests/kalman.test.mjs: con el simulador (una mano sintética quieta, lenta y rápida, con ruido como el
// de MediaPipe) se elige el par que deja menos temblor con la mano quieta sin atrasarse en movimiento más de
// 1,75 veces el ruido original.
export const PARAMETROS = { r: 0.002 ** 2, q: 0.01, adaptacion: 200, saltoMaximo: 0.15 };

function crearEscalar(x0) {
  let x = x0;
  let v = 0;
  let p00 = 1e-4;
  let p01 = 0;
  let p11 = 1e-2;
  // Devuelve la posición estimada y la innovación normalizada (innovación² / su varianza esperada)
  return function paso(medida, dt, r, q) {
    // predicción
    x += v * dt;
    const dt2 = dt * dt;
    const n00 = p00 + 2 * dt * p01 + dt2 * p11 + (q * dt2 * dt2) / 4;
    const n01 = p01 + dt * p11 + (q * dt2 * dt) / 2;
    const n11 = p11 + q * dt2;
    // corrección (se mide solo la posición)
    const s = n00 + r;
    const k0 = n00 / s;
    const k1 = n01 / s;
    const innovacion = medida - x;
    x += k0 * innovacion;
    v += k1 * innovacion;
    p00 = (1 - k0) * n00;
    p01 = (1 - k0) * n01;
    p11 = n11 - k1 * n01;
    return [x, (innovacion * innovacion) / s];
  };
}

// Kalman adaptativo por mano: si las innovaciones de los 63 filtros son, en promedio, más grandes de lo que el
// modelo espera, la mano aceleró (una "maniobra") y se sube q para seguirla; con la mano quieta, q vuelve al
// mínimo y suaviza. El promedio de 63 innovaciones normalizadas es un estadístico chi-cuadrado (63 grados de
// libertad) dividido por 63: vale ≈ 1 cuando el modelo describe bien el movimiento y casi no fluctúa.
function crearMano(puntos) {
  const filtros = puntos.map((p) => ['x', 'y', 'z'].map((e) => crearEscalar(p[e])));
  let nis = 1;
  return (puntos2, dt, { r, q, adaptacion }) => {
    const qEfectivo = q * (1 + adaptacion * Math.max(0, nis - 1));
    let suma = 0;
    const salida = puntos2.map((p, i) => {
      const [x, nx] = filtros[i][0](p.x, dt, r, qEfectivo);
      const [y, ny] = filtros[i][1](p.y, dt, r, qEfectivo);
      const [z, nz] = filtros[i][2](p.z, dt, r, qEfectivo);
      suma += nx + ny + nz;
      return { x, y, z };
    });
    nis = suma / (3 * puntos2.length);
    return salida;
  };
}

/**
 * Crea el filtro. filtrar(manos, lados, tMs) devuelve las manos suavizadas (mismo formato que MediaPipe).
 * Cada mano se sigue por su lado (derecha o izquierda); si aparece, desaparece o salta lejos (otra mano que
 * entró al cuadro), su filtro arranca de nuevo para no arrastrar la posición vieja.
 */
export function crearFiltroManos(parametros = PARAMETROS) {
  let anteriores = new Map();
  let tAnterior = null;
  return {
    parametros,
    filtrar(manos, lados = [], tMs = performance.now()) {
      const dt = tAnterior === null ? 1 / 30 : Math.min(Math.max((tMs - tAnterior) / 1000, 1 / 240), 0.25);
      tAnterior = tMs;
      const nuevos = new Map();
      const salida = manos.map((puntos, i) => {
        const clave = lados[i] ?? `mano${i}`;
        const previa = anteriores.get(clave);
        const salto = previa && Math.hypot(puntos[0].x - previa.muneca.x, puntos[0].y - previa.muneca.y);
        const filtro = previa && salto < parametros.saltoMaximo ? previa.filtro : crearMano(puntos);
        const suaves = filtro(puntos, dt, parametros);
        nuevos.set(clave, { filtro, muneca: puntos[0] });
        return suaves;
      });
      anteriores = nuevos;
      return salida;
    },
    reiniciar() {
      anteriores = new Map();
      tAnterior = null;
    },
  };
}
