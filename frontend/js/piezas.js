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

// ---------------------------------------------------------------- qué nodos del .glb son piezas
//
// Los modelos propios tienen un nodo por pieza directamente bajo la escena. Los de Sketchfab (Objaverse) tienen un
// único nodo raíz y las mallas cuelgan varios niveles más abajo: se baja por el árbol hasta un nivel que reparta las
// mallas, respetando cómo las agrupó el autor. Misma regla en Python: contar_piezas de ml/catalogo3d/indexar.py.

export const MAXIMO_PIEZAS = 80;   // más que eso, el despiece no se lee: se agrupan por cercanía
const CUOTA_DOMINANTE = 0.9;       // un nodo con más de esta parte de las mallas del nivel se abre en sus hijos

/**
 * Nodos que son piezas. hijos(nodo): sus nodos hijos · esMalla(nodo): si el nodo mismo tiene malla.
 * Desde los hijos de la raíz, mientras un nodo sin malla propia concentre más del 90 % de las mallas del nivel
 * (por ejemplo, el único hijo de una cadena), se lo reemplaza por sus hijos. Los nodos sin mallas no cuentan.
 */
export function nivelDePiezas(raiz, hijos, esMalla) {
  const mallas = new Map();
  const contar = (n) => {
    if (!mallas.has(n)) mallas.set(n, (esMalla(n) ? 1 : 0) + hijos(n).reduce((s, h) => s + contar(h), 0));
    return mallas.get(n);
  };
  const conMallas = (n) => hijos(n).filter((h) => contar(h) > 0);
  let nivel = conMallas(raiz);
  for (;;) {
    const total = nivel.reduce((s, n) => s + contar(n), 0);
    const k = nivel.findIndex((n) => !esMalla(n) && contar(n) > CUOTA_DOMINANTE * total);
    if (k < 0) return nivel;
    nivel = [...nivel.slice(0, k), ...conMallas(nivel[k]), ...nivel.slice(k + 1)];
  }
}

const distancia2 = (a, b) => a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0);

/**
 * Agrupa piezas por cercanía de sus centros en `maximo` grupos como mucho (k-medias, arrancando por los puntos más
 * alejados entre sí para que sea determinístico). Devuelve listas de índices, ordenadas por su primer índice.
 */
export function agruparPorCercania(centros, maximo = MAXIMO_PIEZAS) {
  const n = centros.length;
  if (n <= maximo) return centros.map((_, i) => [i]);
  const media = [0, 1, 2].map((k) => centros.reduce((s, c) => s + c[k], 0) / n);
  // Semillas: la más alejada del centro y después, de a una, la más alejada de las ya elegidas
  const cercania = centros.map((c) => distancia2(c, media));
  const semillas = [];
  for (let j = 0; j < maximo; j++) {
    const i = cercania.reduce((m, d, k) => (d > cercania[m] ? k : m), 0);
    if (cercania[i] === 0 && j > 0) break; // quedan solo puntos repetidos
    semillas.push([...centros[i]]);
    centros.forEach((c, k) => { cercania[k] = j === 0 ? distancia2(c, centros[i]) : Math.min(cercania[k], distancia2(c, centros[i])); });
  }
  let asignacion = new Array(n).fill(-1);
  const iteraciones = n > 5000 ? 15 : 50;   // miles de partes sueltas (follaje, texto): cada vuelta es n · máximo
  for (let iter = 0; iter < iteraciones; iter++) {
    const nueva = centros.map((c) => semillas.reduce((m, s, j) => (distancia2(c, s) < distancia2(c, semillas[m]) ? j : m), 0));
    if (nueva.every((g, i) => g === asignacion[i])) break;
    asignacion = nueva;
    semillas.forEach((s, j) => {
      const miembros = centros.filter((_, i) => asignacion[i] === j);
      if (miembros.length) for (let k = 0; k < 3; k++) s[k] = miembros.reduce((t, c) => t + c[k], 0) / miembros.length;
    });
  }
  const grupos = new Map();
  asignacion.forEach((g, i) => { if (!grupos.has(g)) grupos.set(g, []); grupos.get(g).push(i); });
  return [...grupos.values()];
}

// ---------------------------------------------------------------- modelos de una sola malla
//
// Muchos modelos de Objaverse vienen como una o dos mallas: el autor no los armó por piezas. Para que cualquier
// objeto se pueda desarmar:
//   1. cada malla se separa en sus partes sueltas (componentes conexas: grupos de triángulos que no se tocan entre
//      sí). Muchas mallas "únicas" son en realidad varias piezas que no se tocan: ruedas, tapas, tornillos;
//   2. si con eso quedan menos de MINIMO_PIEZAS, o si una parte tiene casi todo el modelo (un animal con los ojos
//      sueltos: lo único que saldría son los ojos), esa parte se corta en SECCIONES por cercanía (k-medias de los
//      centros de los triángulos), como un despiece por zonas.

export const MINIMO_PIEZAS = 4;
export const SECCIONES = 6;
export const PARTE_DOMINANTE = 0.8;   // fracción de los triángulos a partir de la cual la parte más grande se corta

/** Si hay que cortar la parte más grande (partes: listas de triángulos). */
export function hayQueSeccionar(tamanos) {
  const total = tamanos.reduce((s, n) => s + n, 0);
  return tamanos.length < MINIMO_PIEZAS || Math.max(...tamanos) > PARTE_DOMINANTE * total;
}
const CELDAS = 131072;        // 2^17 por eje: la clave del vértice (x + y·2^17 + z·2^34) entra justa en 2^51
const MUESTRA_SECCIONES = 20000;

/** Índice del vértice de la esquina c del triángulo t (con o sin buffer de índices). */
const vertice = (indices, t, c) => (indices ? indices[3 * t + c] : 3 * t + c);

/**
 * Partes sueltas de una malla. posiciones: xyz planos · indices: los de la malla o null si no tiene.
 * Los vértices en la misma posición se unen antes: los glTF duplican vértices en las costuras de la textura, y sin
 * unirlos una pieza se partiría por cada costura. Devuelve listas de triángulos, de la parte más grande a la más chica.
 */
export function componentesConexas(posiciones, indices) {
  const nV = posiciones.length / 3;
  const nT = (indices ? indices.length : nV) / 3;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nV; i++) {
    for (let k = 0; k < 3; k++) {
      const v = posiciones[3 * i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  const escala = max.map((m, k) => (CELDAS - 1) / Math.max(m - min[k], 1e-12));
  // Vértice -> representante (el primero en esa posición)
  const unico = new Int32Array(nV);
  const vistos = new Map();
  for (let i = 0; i < nV; i++) {
    const clave = Math.round((posiciones[3 * i] - min[0]) * escala[0])
      + Math.round((posiciones[3 * i + 1] - min[1]) * escala[1]) * CELDAS
      + Math.round((posiciones[3 * i + 2] - min[2]) * escala[2]) * CELDAS * CELDAS;
    const previo = vistos.get(clave);
    if (previo === undefined) vistos.set(clave, i);
    unico[i] = previo ?? i;
  }
  // Unión-búsqueda sobre los vértices: los tres de cada triángulo quedan en el mismo conjunto
  const padre = Int32Array.from({ length: nV }, (_, i) => i);
  const raiz = (i) => {
    while (padre[i] !== i) {
      padre[i] = padre[padre[i]];
      i = padre[i];
    }
    return i;
  };
  for (let t = 0; t < nT; t++) {
    const a = raiz(unico[vertice(indices, t, 0)]);
    for (const c of [1, 2]) {
      const b = raiz(unico[vertice(indices, t, c)]);
      if (a !== b) padre[b] = a;
    }
  }
  const partes = new Map();
  for (let t = 0; t < nT; t++) {
    const r = raiz(unico[vertice(indices, t, 0)]);
    if (!partes.has(r)) partes.set(r, []);
    partes.get(r).push(t);
  }
  return [...partes.values()].sort((a, b) => b.length - a.length);
}

/** Centro (promedio de los centros de sus triángulos) de una lista de triángulos. */
export function centroTriangulos(posiciones, indices, triangulos) {
  const suma = [0, 0, 0];
  for (const t of triangulos) {
    for (let c = 0; c < 3; c++) {
      const v = vertice(indices, t, c);
      for (let k = 0; k < 3; k++) suma[k] += posiciones[3 * v + k];
    }
  }
  return suma.map((s) => s / (3 * triangulos.length || 1));
}

/**
 * Corta una parte (lista de triángulos) en k secciones por cercanía. Las semillas se eligen por puntos más
 * alejados y las medias se ajustan con una muestra de hasta 20 mil triángulos (determinístico); después cada
 * triángulo va a la sección más cercana. Devuelve las secciones no vacías.
 */
export function seccionar(posiciones, indices, triangulos, k = SECCIONES) {
  const centro = (t) => {
    const s = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const v = vertice(indices, t, c);
      for (let j = 0; j < 3; j++) s[j] += posiciones[3 * v + j] / 3;
    }
    return s;
  };
  const paso = Math.max(1, Math.floor(triangulos.length / MUESTRA_SECCIONES));
  const muestra = [];
  for (let i = 0; i < triangulos.length; i += paso) muestra.push(centro(triangulos[i]));
  const grupos = agruparPorCercania(muestra, k);
  const medias = grupos.map((g) => [0, 1, 2].map((j) => g.reduce((s, i) => s + muestra[i][j], 0) / g.length));
  const secciones = medias.map(() => []);
  for (const t of triangulos) {
    const c = centro(t);
    let mejor = 0;
    medias.forEach((m, j) => { if (distancia2(c, m) < distancia2(c, medias[mejor])) mejor = j; });
    secciones[mejor].push(t);
  }
  return secciones.filter((s) => s.length);
}

/**
 * Nombres según dónde está cada pieza respecto del centro del modelo ("Sección superior izquierda"), para poder
 * pedirlas por voz. Y es arriba y X es derecha (convención de glTF); Z positivo mira a la cámara. Si dos quedan
 * con el mismo nombre, se numeran. prefijo: uno para todas o uno por pieza ("Pieza", "Sección").
 */
export function nombresPorPosicion(centros, centroModelo, tamano, prefijo = 'Pieza') {
  const umbral = tamano * 0.08;
  const lado = (v, positivo, negativo) => (v > umbral ? positivo : v < -umbral ? negativo : null);
  const base = centros.map((c, i) => {
    const pre = Array.isArray(prefijo) ? prefijo[i] : prefijo;
    const [x, y, z] = restar(c, centroModelo);
    const partes = [lado(y, 'superior', 'inferior'), lado(z, 'delantera', 'trasera'), lado(x, 'derecha', 'izquierda')]
      .filter(Boolean);
    return `${pre} ${partes.length ? partes.join(' ') : 'central'}`;
  });
  const total = new Map();
  base.forEach((n) => total.set(n, (total.get(n) ?? 0) + 1));
  const usados = new Map();
  return base.map((n) => {
    if (total.get(n) === 1) return n;
    usados.set(n, (usados.get(n) ?? 0) + 1);
    return `${n} ${usados.get(n)}`;
  });
}

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
    p.radial = !unitario(p.direccion ?? []); // sin dirección declarada ni planificada: sale desde el centro
    const desdeCentro = restar(p.centro, centroModelo);
    p.dir = unitario(p.direccion ?? []) ?? unitario(desdeCentro) ?? [0, 1, 0];
    // Las piezas más alejadas en su dirección se alejan más: así no quedan superpuestas
    p.dist = tamano * 0.35 + Math.max(0, punto(desdeCentro, p.dir));
  }

  // Entre piezas que salen en la misma dirección, la que se saca antes queda más afuera
  // (como en un despiece técnico: los bulones por fuera de la brida que sujetan). Las que salen desde el centro
  // ya quedan separadas por su distancia al centro: escalonarlas, con decenas de piezas, las mandaría lejísimos.
  const desmontables = piezas.filter((p) => !p.fija && !p.radial).sort((a, b) => a.orden - b.orden);
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

// ---------------------------------------------------------------- nombres que no dicen qué es la pieza

// Palabras de los nombres que ponen los programas de 3D (o el visor) y no dicen qué es la pieza
const GENERICAS = new Set(['pieza', 'seccion', 'grupo', 'partes', 'parte', 'part', 'object', 'obj', 'objeto', 'mesh',
  'malla', 'node', 'nodo', 'group', 'cube', 'cylinder', 'sphere', 'plane', 'torus', 'cone', 'circle', 'polysurface',
  'pcube', 'pcylinder', 'psphere', 'geometry', 'geom', 'default', 'body', 'solid', 'component', 'untitled', 'model',
  'root', 'scene', 'material', 'mat', 'shape', 'instance', 'primitive', 'box', 'superior', 'inferior', 'delantera',
  'trasera', 'derecha', 'izquierda', 'central', 'sketchfab', 'gltf', 'defaultmaterial', 'lambert', 'blinn', 'phong',
  'cuerpo', 'body', 'brep', 'face', 'surface', 'nurbs', 'curve', 'merged', 'combined', 'join', 'joined']);

/**
 * true si el nombre de una pieza no dice qué es: los que pone el visor ("Sección superior izquierda 2",
 * "Grupo 3 (12 partes)") o los programas de 3D ("Object_12", "Cylinder.004", "polySurface37"). Esas son las que
 * puede nombrar el reconocimiento de piezas; un nombre puesto por el autor ("Piston", "Carburetor") se respeta.
 */
export function nombreGenerico(nombre = '') {
  const palabras = nombre.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()
    .replace(/([a-z])(\d)/g, '$1 $2').split(/[^a-z]+/).filter(Boolean);
  return palabras.every((p) => GENERICAS.has(p) || p.length <= 1);
}
