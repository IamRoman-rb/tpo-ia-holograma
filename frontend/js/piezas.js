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
    // Sin orden (por ejemplo, un modelo generado sin metadatos): queda fija la pieza más grande. Si se conocen
    // las cajas de las piezas, el orden y la dirección de salida los busca A* (planificarDesarme); si no,
    // salen desde la más externa hacia adentro.
    const mayor = piezas.reduce((a, b) => ((b.volumen ?? 0) > (a.volumen ?? 0) ? b : a));
    const resto = piezas.filter((p) => p !== mayor);
    const plan = piezas.every((p) => p.tamano) ? planificarDesarme(piezas, piezas.indexOf(mayor)) : null;
    if (plan) {
      plan.pasos.forEach(({ pieza, dir }, k) => {
        piezas[pieza].orden = k + 1;
        piezas[pieza].direccion ??= dir;
      });
    } else {
      resto
        .sort((a, b) => largo(restar(b.centro, centroModelo)) - largo(restar(a.centro, centroModelo)))
        .forEach((p, i) => { p.orden = i + 1; });
    }
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

// ---------------------------------------------------------------- planificación del desarme con A*
//
// Problema clásico de planificación de ensambles: ¿en qué orden y hacia dónde se saca cada pieza sin que choque
// con las que siguen puestas? Se resuelve con búsqueda A* sobre los estados "qué piezas ya salieron":
//   - una pieza puede salir en una dirección si la caja que barre al deslizarse no toca ninguna de las que quedan;
//   - costo de un paso: 1, más 0,5 si cambia la dirección respecto del paso anterior (dar vuelta el objeto
//     cuesta), más lo que recorre la pieza hasta salir del modelo (en tamaños del modelo: las de afuera, antes);
//   - heurística: la cantidad de piezas que faltan (cada paso cuesta al menos 1). Nunca sobreestima, así que
//     A* devuelve el plan de menor costo.
// Si las piezas están trabadas (ninguna puede salir sin chocar), se permite sacar la que menos choca, con un
// costo alto: el plan siempre termina.

export const DIRECCIONES = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]];
const MARGEN = 0.02;          // las caras que se tocan no cuentan como choque (piezas apoyadas)
const COSTO_GIRO = 0.5;
const COSTO_TRABADA = 10;
const MAXIMO_EXPANSIONES = 20000;

// Eje en el que más avanza una dirección (las diagonales se tratan como su eje dominante)
const ejePrincipal = (d) => [0, 1, 2].reduce((m, k) => (Math.abs(d[k]) > Math.abs(d[m]) ? k : m), 0);

function caja(p) {
  const m = p.tamano.map((t) => (t / 2) * (1 - MARGEN));
  return { min: p.centro.map((c, k) => c - m[k]), max: p.centro.map((c, k) => c + m[k]) };
}

/** ¿La caja a, deslizándose en la dirección d (unitaria y sobre un eje), choca con la caja b? */
function barrido(a, b, d) {
  const eje = ejePrincipal(d);
  for (let k = 0; k < 3; k++) {
    if (k !== eje && (a.max[k] <= b.min[k] || b.max[k] <= a.min[k])) return false;
  }
  // En el eje de salida, b está en el camino si sobresale por delante de a. Una pieza metida en otra pero con la
  // punta afuera (un tornillo en la base) puede salir; una encerrada (un eje dentro de la carcasa), no.
  return d[eje] > 0 ? b.max[eje] > a.max[eje] : b.min[eje] < a.min[eje];
}

/**
 * piezas: [{ centro, tamano, direccion? }] · fija: índice de la pieza que no se saca.
 * Devuelve { pasos: [{ pieza, dir, trabada }], costo, expansiones } o null si hay demasiadas piezas.
 */
export function planificarDesarme(piezas, fija = -1) {
  const n = piezas.length;
  if (n > 24) return null;
  const cajas = piezas.map(caja);
  const minimo = [0, 1, 2].map((k) => Math.min(...cajas.map((c) => c.min[k])));
  const maximo = [0, 1, 2].map((k) => Math.max(...cajas.map((c) => c.max[k])));
  const tamanoModelo = largo(restar(maximo, minimo)) || 1;
  // Lo que recorre la pieza hasta salir del modelo, en tamaños del modelo (las de afuera salen antes)
  const recorrido = (i, d) => {
    const eje = ejePrincipal(d);
    return (d[eje] > 0 ? maximo[eje] - cajas[i].min[eje] : cajas[i].max[eje] - minimo[eje]) / tamanoModelo;
  };
  const opciones = piezas.map((p) => (unitario(p.direccion ?? []) ? [unitario(p.direccion)] : DIRECCIONES));
  const removibles = piezas.map((_, i) => i).filter((i) => i !== fija);
  const todas = removibles.reduce((m, i) => m | (1 << i), 0);

  // Movimientos posibles desde un estado: cada pieza que queda, en cada dirección; si ninguno sale libre,
  // la pieza y dirección que menos choca, marcada como trabada
  function movimientos(sacadas) {
    const libres = [];
    let menosTrabada = null;
    for (const i of removibles) {
      if (sacadas & (1 << i)) continue;
      for (const d of opciones[i]) {
        let choques = 0;
        for (let j = 0; j < n; j++) {
          if (j === i || (sacadas & (1 << j))) continue;
          if (barrido(cajas[i], cajas[j], d)) choques++;
        }
        const m = { pieza: i, dir: d, recorrido: recorrido(i, d), trabada: choques > 0, choques };
        if (!choques) libres.push(m);
        else if (!menosTrabada || choques < menosTrabada.choques) menosTrabada = m;
      }
    }
    return libres.length ? libres : menosTrabada ? [menosTrabada] : [];
  }

  const restantes = (sacadas) => removibles.filter((i) => !(sacadas & (1 << i))).length;
  // Cola de prioridad (montículo binario) ordenada por f = g + h
  const monticulo = [];
  const subir = (k) => {
    while (k > 0) {
      const padre = (k - 1) >> 1;
      if (monticulo[padre].f <= monticulo[k].f) break;
      [monticulo[padre], monticulo[k]] = [monticulo[k], monticulo[padre]];
      k = padre;
    }
  };
  const bajar = (k) => {
    for (;;) {
      const a = 2 * k + 1;
      const b = a + 1;
      let menor = k;
      if (a < monticulo.length && monticulo[a].f < monticulo[menor].f) menor = a;
      if (b < monticulo.length && monticulo[b].f < monticulo[menor].f) menor = b;
      if (menor === k) return;
      [monticulo[menor], monticulo[k]] = [monticulo[k], monticulo[menor]];
      k = menor;
    }
  };
  const meter = (nodo) => { monticulo.push(nodo); subir(monticulo.length - 1); };
  const sacar = () => {
    const tope = monticulo[0];
    const ultimo = monticulo.pop();
    if (monticulo.length) { monticulo[0] = ultimo; bajar(0); }
    return tope;
  };

  // El estado incluye la última dirección (el costo del giro depende de ella)
  const clave = (sacadas, dir) => `${sacadas}|${dir ? dir.join() : ''}`;
  const mejorCosto = new Map([[clave(0, null), 0]]);
  meter({ sacadas: 0, dir: null, g: 0, f: restantes(0), pasos: [] });
  let expansiones = 0;
  while (monticulo.length) {
    const nodo = sacar();
    if (nodo.sacadas === todas) return { pasos: nodo.pasos, costo: nodo.g, expansiones };
    if (nodo.g > (mejorCosto.get(clave(nodo.sacadas, nodo.dir)) ?? Infinity)) continue;
    if (++expansiones > MAXIMO_EXPANSIONES) break;
    for (const m of movimientos(nodo.sacadas)) {
      const giro = nodo.dir && m.dir.join() !== nodo.dir.join() ? COSTO_GIRO : 0;
      const g = nodo.g + 1 + giro + m.recorrido + (m.trabada ? COSTO_TRABADA * m.choques : 0);
      const sacadas = nodo.sacadas | (1 << m.pieza);
      const k = clave(sacadas, m.dir);
      if (g >= (mejorCosto.get(k) ?? Infinity)) continue;
      mejorCosto.set(k, g);
      meter({ sacadas, dir: m.dir, g, f: g + restantes(sacadas),
        pasos: [...nodo.pasos, { pieza: m.pieza, dir: m.dir, trabada: m.trabada }] });
    }
  }
  return null;
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

/**
 * Encuadre de una pieza aislada para analizarla: escala y posición del contenedor del modelo para
 * que la pieza quede centrada y ocupe la vista como lo hacía el objeto entero.
 * centro: centro de la pieza en coordenadas del contenedor; radio: radio de la pieza (mismas unidades).
 */
export function encuadreAislada(centro, radio, escalaBase, radioArmado) {
  const escala = escalaBase * (radioArmado / Math.max(radio, radioArmado * 0.05)) * 0.8;
  return { escala, posicion: centro.map((v) => -v * escala) };
}

const MATERIALES = { acero: 'Acero', hierro: 'Hierro', aluminio: 'Aluminio', laton: 'Latón', cobre: 'Cobre',
  plastico: 'Plástico', goma: 'Goma', pintado: 'Metal pintado' };

/** Nombre legible de un material del .glb ("acero_ff0000" → "Acero"), o null si no es un material conocido. */
export function nombreMaterial(nombre = '') {
  return MATERIALES[nombre.split('_')[0]] ?? null;
}

/** "12,5 × 4 × 4 cm" (con unidad) o proporciones relativas a la mayor medida si el modelo no la declara. */
export function textoMedidas(tamano, unidad) {
  const fmt = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1)).replace('.', ',');
  if (unidad) return `${tamano.map(fmt).join(' × ')} ${unidad}`;
  const mayor = Math.max(...tamano) || 1;
  return `${tamano.map((v) => fmt((100 * v) / mayor)).join(' × ')} (proporciones; la mayor = 100)`;
}

/**
 * Selección por permanencia: apuntar a la misma pieza durante `ms` la selecciona (patrón habitual en
 * interfaces de gestos, donde no hay clic). actualizar(id, t) devuelve { progreso: 0-1, elegida: id | null };
 * elegida vale el id solo en el instante en que se completa.
 */
export function crearSeleccionPorPermanencia(ms = 800) {
  let actual = null;
  let desde = 0;
  let hecha = false;
  return function actualizar(id, t) {
    if (id !== actual) {
      actual = id;
      desde = t;
      hecha = false;
    }
    if (!id) return { progreso: 0, elegida: null };
    const progreso = Math.min(1, (t - desde) / ms);
    if (progreso >= 1 && !hecha) {
      hecha = true;
      return { progreso, elegida: id };
    }
    return { progreso, elegida: null };
  };
}
