// Reconocimiento de piezas de motores por su forma, en el navegador.
//
// Los modelos de Objaverse se desarman en el visor, pero sus piezas se llaman por su lugar ("Sección superior
// izquierda"). Esta red (entrenada en ml/piezas/entrenar_piezas.py con motores generados por código) mira la forma
// de cada pieza y su lugar en el conjunto, y dice qué pieza de motor es. Solo se usa con modelos de motores, y
// solo se nombra una pieza si la red está segura: si no, conserva el nombre por su lugar.
//
// Los rasgos son los mismos que en Python (ml/piezas/rasgos.py), cuenta por cuenta: frontend/tests/
// reconocer_piezas.test.mjs lo verifica con los números que exporta el entrenamiento.

export const URL_RED = 'ml/piezas_motor.json';

// ---------------------------------------------------------------- pesos

function decodificar({ forma, float32 }) {
  const binario = typeof atob === 'function' ? atob(float32) : Buffer.from(float32, 'base64').toString('binary');
  const bytes = new Uint8Array(binario.length);
  for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
  return { datos: new Float32Array(bytes.buffer), forma };
}

/** Prepara la red exportada (JSON) para usarla. */
export function leerRed(json) {
  const capa = ({ W, b }) => {
    const w = decodificar(W);
    return { W: w.datos, entradas: w.forma[0], salidas: w.forma[1], b: decodificar(b).datos };
  };
  return {
    ...json,
    media: decodificar(json.media_contexto).datos,
    desvio: decodificar(json.desvio_contexto).datos,
    punto: json.capas_punto.map(capa),
    cabeza: json.capas_cabeza.map(capa),
  };
}

let cargada = null;
export function cargarRed(url = URL_RED) {
  cargada ??= fetch(url).then((r) => {
    if (!r.ok) throw new Error(`No se pudo cargar ${url} (${r.status})`);
    return r.json();
  }).then(leerRed).catch((e) => {
    cargada = null;
    throw e;
  });
  return cargada;
}

// ---------------------------------------------------------------- muestreo de la superficie

/**
 * n puntos al azar sobre la superficie de una pieza (proporcional al área de cada triángulo), en coordenadas
 * del modelo. fuentes: [{ xyz: posiciones planas, indices (o null), matriz: 16 valores de una Matrix4 de three
 * (de la malla al modelo) }]. azar(): número en [0, 1). Devuelve { puntos: Float64Array(n·3), area }.
 */
export function muestrearPieza(fuentes, n, azar) {
  const tri = [];        // por fuente: áreas de sus triángulos (en el modelo, con la escala de la malla)
  let total = 0;
  const v = (f, i, salida) => {
    const e = f.matriz;
    const x = f.xyz[i * 3];
    const y = f.xyz[i * 3 + 1];
    const z = f.xyz[i * 3 + 2];
    salida[0] = e[0] * x + e[4] * y + e[8] * z + e[12];
    salida[1] = e[1] * x + e[5] * y + e[9] * z + e[13];
    salida[2] = e[2] * x + e[6] * y + e[10] * z + e[14];
    return salida;
  };
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  const c = [0, 0, 0];
  for (const f of fuentes) {
    const cantidad = f.indices ? f.indices.length / 3 : f.xyz.length / 9;
    const areas = new Float64Array(cantidad);
    for (let t = 0; t < cantidad; t++) {
      const [i0, i1, i2] = f.indices ? [f.indices[t * 3], f.indices[t * 3 + 1], f.indices[t * 3 + 2]] : [t * 3, t * 3 + 1, t * 3 + 2];
      v(f, i0, a); v(f, i1, b); v(f, i2, c);
      const ux = b[0] - a[0]; const uy = b[1] - a[1]; const uz = b[2] - a[2];
      const wx = c[0] - a[0]; const wy = c[1] - a[1]; const wz = c[2] - a[2];
      const area = 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
      areas[t] = Number.isFinite(area) ? area : 0;
      total += areas[t];
    }
    tri.push(areas);
  }
  const puntos = new Float64Array(n * 3);
  if (!(total > 0)) {
    const f = fuentes.find((g) => g.xyz.length);
    if (f) for (let k = 0; k < n; k++) puntos.set(v(f, 0, a), k * 3);
    return { puntos, area: 0 };
  }
  // Áreas acumuladas de todas las fuentes, para elegir triángulos con búsqueda binaria
  const acumulada = new Float64Array(tri.reduce((s, t) => s + t.length, 0));
  const deDonde = [];
  let s = 0;
  let k0 = 0;
  tri.forEach((areas, j) => {
    for (let t = 0; t < areas.length; t++) { s += areas[t]; acumulada[k0 + t] = s; }
    deDonde.push(k0);
    k0 += areas.length;
  });
  for (let k = 0; k < n; k++) {
    const r = azar() * total;
    let lo = 0;
    let hi = acumulada.length - 1;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (acumulada[m] < r) lo = m + 1; else hi = m;
    }
    let j = deDonde.length - 1;
    while (deDonde[j] > lo) j--;
    const f = fuentes[j];
    const t = lo - deDonde[j];
    const [i0, i1, i2] = f.indices ? [f.indices[t * 3], f.indices[t * 3 + 1], f.indices[t * 3 + 2]] : [t * 3, t * 3 + 1, t * 3 + 2];
    v(f, i0, a); v(f, i1, b); v(f, i2, c);
    let u = azar();
    let w = azar();
    if (u + w > 1) { u = 1 - u; w = 1 - w; }
    for (let d = 0; d < 3; d++) puntos[k * 3 + d] = a[d] + u * (b[d] - a[d]) + w * (c[d] - a[d]);
  }
  return { puntos, area: total };
}

// ---------------------------------------------------------------- rasgos (igual que ml/piezas/rasgos.py)

/** Autovalores y autovectores de una matriz simétrica 3×3 (Jacobi). Devuelve { valores, vectores[col] }. */
export function autovalores3(m) {
  const a = m.map((fila) => fila.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let barrido = 0; barrido < 50; barrido++) {
    const fuera = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (fuera < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {          // columnas p y q
        const akp = a[k][p];
        const akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {          // filas p y q
        const apk = a[p][k];
        const aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p];
        const vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return { valores: [a[0][0], a[1][1], a[2][2]], vectores: V };
}

// Centro, autovalores (de mayor a menor, no negativos) y ejes principales (columnas) de una nube de puntos.
// Covarianza con N − 1, como np.cov.
function ejes(puntos) {
  const n = puntos.length / 3;
  const c = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) c[d] += puntos[i * 3 + d];
  for (let d = 0; d < 3; d++) c[d] /= n;
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const x = [puntos[i * 3] - c[0], puntos[i * 3 + 1] - c[1], puntos[i * 3 + 2] - c[2]];
    for (let p = 0; p < 3; p++) for (let q = p; q < 3; q++) cov[p][q] += x[p] * x[q];
  }
  for (let p = 0; p < 3; p++) {
    for (let q = p; q < 3; q++) { cov[p][q] /= Math.max(1, n - 1); cov[q][p] = cov[p][q]; }
    cov[p][p] += 1e-12;
  }
  const { valores, vectores } = autovalores3(cov);
  const orden = [0, 1, 2].sort((i, j) => valores[j] - valores[i]);
  return {
    centro: c,
    valores: orden.map((i) => Math.max(valores[i], 0)),
    ejes: orden.map((i) => [vectores[0][i], vectores[1][i], vectores[2][i]]),   // ejes[k] = k-ésimo eje
  };
}

/**
 * Rasgos de cada pieza a partir de sus puntos ya muestreados y su área (como rasgos_de_muestras en Python).
 * muestras: [Float64Array(n·3)] en coordenadas del modelo. Devuelve { puntos: [Float32Array(n·3)], contexto: [[15]] }.
 */
export function rasgosDeMuestras(muestras, areas) {
  const totalPuntos = muestras.reduce((s, m) => s + m.length, 0);
  const todos = new Float64Array(totalPuntos);
  let o = 0;
  for (const m of muestras) { todos.set(m, o); o += m.length; }
  const global = ejes(todos);
  const escala = Math.sqrt(global.valores[0]) * 2 || 1;
  // Signo de cada eje: que la mayor parte del área quede del lado positivo (tercer momento)
  const E = global.ejes.map((eje) => {
    let m3 = 0;
    for (let i = 0; i < todos.length / 3; i++) {
      const p = (todos[i * 3] - global.centro[0]) * eje[0] + (todos[i * 3 + 1] - global.centro[1]) * eje[1]
        + (todos[i * 3 + 2] - global.centro[2]) * eje[2];
      m3 += p * p * p;
    }
    return m3 < 0 ? eje.map((x) => -x) : eje;
  });
  const areaTotal = areas.reduce((s, a) => s + a, 0) || 1;

  const puntos = [];
  const contexto = [];
  const firmas = [];
  muestras.forEach((m, j) => {
    const n = m.length / 3;
    const local = new Float64Array(m.length);
    for (let i = 0; i < n; i++) {
      const x = m[i * 3] - global.centro[0];
      const y = m[i * 3 + 1] - global.centro[1];
      const z = m[i * 3 + 2] - global.centro[2];
      for (let k = 0; k < 3; k++) local[i * 3 + k] = (x * E[k][0] + y * E[k][1] + z * E[k][2]) / escala;
    }
    const { centro: c, valores: v } = ejes(local);
    const tam = Math.sqrt(v[0]) * 2 || 1e-6;
    const forma = new Float32Array(m.length);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        const val = local[i * 3 + k];
        forma[i * 3 + k] = (val - c[k]) / tam;
        if (val < min[k]) min[k] = val;
        if (val > max[k]) max[k] = val;
      }
    }
    puntos.push(forma);
    const lineal = (v[0] - v[1]) / (v[0] + 1e-12);
    const plano = (v[1] - v[2]) / (v[0] + 1e-12);
    const bola = v[2] / (v[0] + 1e-12);
    const extension = [0, 1, 2].map((k) => max[k] - min[k]).sort((p, q) => q - p);
    contexto.push([...c, Math.hypot(...c), Math.hypot(c[1], c[2]), ...extension, lineal, plano, bola,
      areas[j] / areaTotal, tam, 0, 0]);
    firmas.push([...extension, lineal, plano]);
  });
  // Piezas "gemelas": misma forma y tamaño (pistones, bielas, bujías…)
  firmas.forEach((f, i) => {
    let parecidas = -1;
    for (const g of firmas) {
      let peor = 0;
      for (let k = 0; k < f.length; k++) peor = Math.max(peor, Math.abs(g[k] - f[k]) / (Math.abs(f[k]) + 0.05));
      if (peor < 0.15) parecidas++;
    }
    contexto[i][13] = Math.log1p(parecidas);
    contexto[i][14] = Math.log1p(muestras.length);
  });
  return { puntos, contexto };
}

// ---------------------------------------------------------------- red

function lineal(entrada, { W, b, entradas, salidas }, relu) {
  const salida = new Float32Array(salidas);
  salida.set(b);
  for (let i = 0; i < entradas; i++) {
    const x = entrada[i];
    if (x === 0) continue;
    const fila = i * salidas;
    for (let j = 0; j < salidas; j++) salida[j] += x * W[fila + j];
  }
  if (relu) for (let j = 0; j < salidas; j++) if (salida[j] < 0) salida[j] = 0;
  return salida;
}

/** Probabilidades de cada clase para una pieza (puntos: Float32Array n·3, contexto: [15]). */
export function predecir(red, puntos, contexto) {
  const n = puntos.length / 3;
  const ultima = red.punto[red.punto.length - 1].salidas;
  const maximo = new Float32Array(ultima).fill(-Infinity);
  const suma = new Float32Array(ultima);
  for (let i = 0; i < n; i++) {
    let h = puntos.subarray(i * 3, i * 3 + 3);
    for (const capa of red.punto) h = lineal(h, capa, true);
    for (let j = 0; j < ultima; j++) {
      if (h[j] > maximo[j]) maximo[j] = h[j];
      suma[j] += h[j];
    }
  }
  const entrada = new Float32Array(2 * ultima + contexto.length);
  entrada.set(maximo);
  for (let j = 0; j < ultima; j++) entrada[ultima + j] = suma[j] / n;
  for (let k = 0; k < contexto.length; k++) entrada[2 * ultima + k] = (contexto[k] - red.media[k]) / red.desvio[k];
  let h = entrada;
  red.cabeza.forEach((capa, k) => { h = lineal(h, capa, k < red.cabeza.length - 1); });
  const m = Math.max(...h);
  const e = Array.from(h, (x) => Math.exp(x - m));
  const total = e.reduce((s, x) => s + x, 0);
  return e.map((x) => x / total);
}

// ---------------------------------------------------------------- de las piezas del visor a nombres

/**
 * Clases que puede tener un modelo según su familia: 'combustion', 'electrico' o 'motor' (cualquiera).
 * Así un motor eléctrico no tiene "pistones" aunque una pieza se parezca.
 */
export function clasesPermitidas(red, familia) {
  return red.clases.map((c) => c === red.otro || familia === 'motor' || !familia
    || (red.info[c]?.familias ?? []).includes(familia));
}

/**
 * Decide el nombre de cada pieza. probabilidades: [[por clase]]. centros: [[x, y, z]] (para numerar en orden).
 * renombrables: [bool] (las que ya tienen un nombre de verdad no se tocan, ni cuentan para numerar).
 * Devuelve por pieza null (se queda con su nombre) o { clase, nombre, descripcion, confianza }.
 */
export function nombrar(red, probabilidades, centros,
  { familia = 'motor', umbral = red.umbral_confianza, renombrables = null } = {}) {
  const permitidas = clasesPermitidas(red, familia);
  const elegidas = probabilidades.map((p, i) => {
    if (renombrables && !renombrables[i]) return null;
    const q = p.map((x, k) => (permitidas[k] ? x : 0));
    const total = q.reduce((s, x) => s + x, 0) || 1;
    let mejor = 0;
    for (let k = 1; k < q.length; k++) if (q[k] > q[mejor]) mejor = k;
    const clase = red.clases[mejor];
    const confianza = q[mejor] / total;
    return clase === red.otro || confianza < umbral ? null : { clase, confianza };
  });
  // Un motor tiene un solo cigüeñal, un cárter, una correa de distribución…: si la red ve más de los que puede
  // haber (info.maximo, contado en los motores generados), se queda con los de más confianza
  const vistas = new Map();
  elegidas.map((e, i) => i).filter((i) => elegidas[i])
    .sort((i, j) => elegidas[j].confianza - elegidas[i].confianza)
    .forEach((i) => {
      const { clase } = elegidas[i];
      vistas.set(clase, (vistas.get(clase) ?? 0) + 1);
      if (vistas.get(clase) > (red.info[clase]?.maximo ?? Infinity)) elegidas[i] = null;
    });
  // Las de una misma clase se numeran de izquierda a derecha (y de abajo arriba, si están a la par)
  const porClase = new Map();
  elegidas.forEach((e, i) => {
    if (!e) return;
    if (!porClase.has(e.clase)) porClase.set(e.clase, []);
    porClase.get(e.clase).push(i);
  });
  for (const [clase, indices] of porClase) {
    const { nombre, descripcion } = red.info[clase];
    indices.sort((i, j) => (centros[i][0] - centros[j][0]) || (centros[i][1] - centros[j][1]));
    indices.forEach((i, k) => {
      Object.assign(elegidas[i], { nombre: indices.length > 1 ? `${nombre} ${k + 1}` : nombre, descripcion });
    });
  }
  return elegidas;
}

const pausa = () => new Promise((r) => setTimeout(r, 0));

/**
 * Todo junto: de las mallas de cada pieza a sus nombres. fuentesPorPieza: [[{ xyz, indices, matriz }]].
 * Cede el hilo cada tanto para no congelar la página con modelos grandes.
 */
export async function reconocer(red, fuentesPorPieza, centros,
  { familia = 'motor', azar = Math.random, renombrables = null } = {}) {
  const muestras = [];
  const areas = [];
  for (const fuentes of fuentesPorPieza) {
    const { puntos, area } = muestrearPieza(fuentes, red.puntos, azar);
    muestras.push(puntos);
    areas.push(area);
    if (muestras.length % 8 === 0) await pausa();
  }
  const { puntos, contexto } = rasgosDeMuestras(muestras, areas);
  const probabilidades = [];
  for (let i = 0; i < puntos.length; i++) {
    probabilidades.push(predecir(red, puntos[i], contexto[i]));
    if (i % 8 === 7) await pausa();
  }
  return nombrar(red, probabilidades, centros, { familia, renombrables });
}
