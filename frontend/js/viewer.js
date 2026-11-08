// Escena Three.js: carga de .glb, transformaciones suavizadas del objeto y desmontaje por piezas.
// Las piezas se pueden apuntar (cursor de la mano), mover de a una y aislar para analizarlas.
// Dos estilos: "solido" (materiales del .glb) y "holograma" (cada pieza en un color neón, con aristas,
// nube de puntos y relleno tenue, pensado para verse sobre el video de la cámara).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { crearHistorial } from './historial.js';
import { reconocer } from './reconocer_piezas.js';
import { areaSuperficie, colorPieza, crearAzar, puntosEnSuperficie } from './holograma.js';
import {
  MAXIMO_PIEZAS, MINIMO_PIEZAS, SECCIONES, agruparPorCercania, centroTriangulos, componentesConexas, desplazamiento,
  encuadreAislada, hayQueSeccionar, nivelDePiezas, nombreGenerico, nombreMaterial, nombresPorPosicion, prepararPiezas,
  seccionar,
  siguienteParaDesmontar, textoMedidas, ultimaDesmontada,
} from './piezas.js';

export const ESCALA_MIN = 0.2;   // mismos límites que backend/safety.py
export const ESCALA_MAX = 5;
const DISTANCIA_CAMARA = 4;
// Navegación con el mouse: la cámara orbita alrededor de un centro; se acerca o se aleja (rueda),
// se desplaza (arrastre) y gira (arrastre con la rueda apretada)
const DISTANCIA_MIN = 0.6;
const DISTANCIA_MAX = 25;
const LIMITE_DESPLAZAMIENTO = 20; // hasta dónde se puede alejar el centro de la vista, para no perderse
const ELEVACION_MIN = -0.25;      // apenas por debajo del horizonte: más abajo se vería la grilla desde atrás
const ELEVACION_MAX = 1.45;       // casi desde arriba (83°), sin dar vuelta la cámara
const RAPIDEZ_SUAVIZADO = 14;   // más alto = sigue a la mano más rápido
const RAPIDEZ_PIEZAS = 6;       // el desmontaje es más lento, para que se vea el recorrido
const COLOR_SELECCION = new THREE.Color(0x2bd4ff);
const NEGRO = new THREE.Color(0x000000);
const OPACIDAD_ATENUADA = 0.08; // las demás piezas mientras se analiza una
const PUNTOS_POR_MODELO = 14000; // nube de puntos del estilo holograma, repartida según el área de cada malla
const OPACIDAD_HOLO = { relleno: 0.1, aristas: 0.7, puntos: 0.9 };

// Textura de un punto redondo y difuso: con mezcla aditiva, la nube de puntos brilla
function texturaPunto() {
  const lienzo = document.createElement('canvas');
  lienzo.width = lienzo.height = 32;
  const ctx = lienzo.getContext('2d');
  const g = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.6)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 32, 32);
  return new THREE.CanvasTexture(lienzo);
}

/**
 * alCambiarPiezas(lista) se llama cada vez que cambia el estado de las piezas.
 * alCuadro({ piezas, escala }) se llama en cada cuadro con dónde se ve cada pieza en la pantalla (para las etiquetas).
 */
export function crearVisor(contenedor, { alCambiarPiezas = () => {}, alCuadro = null, estilo: estiloInicial = 'solido' } = {}) {
  const escena = new THREE.Scene();
  const camara = new THREE.PerspectiveCamera(50, contenedor.clientWidth / contenedor.clientHeight, 0.1, 100);
  // Vista de la cámara: mira al centro desde una distancia y dos ángulos. La actual sigue al objetivo
  // con suavizado, como el pivote. Al empezar: de frente y un poco desde arriba, como antes.
  const vistaInicial = {
    centro: new THREE.Vector3(), distancia: Math.hypot(DISTANCIA_CAMARA, 0.3),
    azimut: 0, elevacion: Math.atan2(0.3, DISTANCIA_CAMARA),
  };
  const copiarVista = (v) => ({ ...v, centro: v.centro.clone() });
  const vistaObjetivo = copiarVista(vistaInicial);
  const vistaActual = copiarVista(vistaInicial);
  // Cámara auxiliar ubicada en la vista objetivo: los cálculos del mouse se hacen donde va a quedar la cámara
  const camaraGuia = camara.clone();
  ubicarCamara(camara, vistaActual);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(contenedor.clientWidth, contenedor.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  contenedor.appendChild(renderer.domElement);

  // Cursor de la mano: sigue la punta del índice al señalar. El aro muestra el avance de la selección.
  const cursor = document.createElement('div');
  cursor.className = 'cursor-mano';
  cursor.hidden = true;
  contenedor.appendChild(cursor);

  // Iluminación: entorno para los reflejos metálicos + contraluz cian de "holograma"
  const pmrem = new THREE.PMREMGenerator(renderer);
  escena.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const contraluz = new THREE.DirectionalLight(0x5ee7ff, 2.5);
  contraluz.position.set(-3, 2, -4);
  escena.add(contraluz);

  const grilla = new THREE.GridHelper(40, 160, 0x2bd4ff, 0x16475a); // amplia: se puede recorrer con el mouse
  grilla.position.y = -1.3;
  grilla.material.transparent = true;
  grilla.material.opacity = 0.35;
  escena.add(grilla);

  let estilo = estiloInicial;
  const puntoRedondo = texturaPunto();

  // El pivote es lo que se manipula; el modelo cargado vive adentro, centrado.
  const pivote = new THREE.Group();
  escena.add(pivote);

  // Estado objetivo: los gestos lo modifican y el pivote lo sigue con suavizado.
  const objetivo = { pos: new THREE.Vector3(), rotX: 0, rotY: 0, escala: 1 };
  // Inercia del giro (rad/s): al soltar las dos palmas girando rápido, el objeto sigue y frena de a poco
  const giro = { x: 0, y: 0 };
  const FRENADO_GIRO_S = 0.9;    // constante de tiempo del frenado
  const detenerGiro = () => { giro.x = 0; giro.y = 0; };

  // Piezas del modelo actual: { id, nombre, descripcion, fija, dir, dist, obj, base, actual, libre, ... }
  let piezas = [];
  let separacion = null;         // si las piezas las separó el visor: 'partes' (sueltas) o 'secciones' (cortes)
  let objetivosPiezas = [];      // 0 = montada · 1 = desmontada (puede ser intermedio: vista explotada)
  let seleccionada = null;
  let apuntada = null;           // pieza bajo el cursor de la mano
  let aislada = null;            // pieza que se está analizando sola
  let unidad = null;             // unidad del modelo, si el .glb la declara (los generados: cm)
  // Encuadre: al desarmar, el modelo se achica para que todas las piezas sigan a la vista
  let modelo = null;
  let contenedorModelo = null;
  let escalaBase = 1;
  let radioArmado = 1;
  let centroModelo = [0, 0, 0];

  const historial = crearHistorial();
  const estadoActual = () => ({
    pos: objetivo.pos.toArray().map((v) => +v.toFixed(4)),
    rotX: +objetivo.rotX.toFixed(4), rotY: +objetivo.rotY.toFixed(4), escala: +objetivo.escala.toFixed(4),
    piezas: objetivosPiezas.map((v) => +v.toFixed(3)),
    libres: piezas.map((p) => p.libre.toArray().map((v) => +v.toFixed(4))),
    aislada,
  });
  const loader = new GLTFLoader();
  const reloj = new THREE.Clock();
  const raycaster = new THREE.Raycaster();

  function ubicarCamara(cam, v) {
    const c = Math.cos(v.elevacion);
    cam.position.set(Math.sin(v.azimut) * c, Math.sin(v.elevacion), Math.cos(v.azimut) * c)
      .multiplyScalar(v.distancia).add(v.centro);
    cam.lookAt(v.centro);
    cam.updateMatrixWorld();
  }

  function guiaEnObjetivo() {
    camaraGuia.projectionMatrix.copy(camara.projectionMatrix);
    camaraGuia.projectionMatrixInverse.copy(camara.projectionMatrixInverse);
    ubicarCamara(camaraGuia, vistaObjetivo);
    return camaraGuia;
  }

  // Derecha y arriba de la pantalla, en coordenadas del mundo
  function ejesPantalla(cam = camara) {
    return {
      derecha: new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0),
      arriba: new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1),
    };
  }

  function limitarVista() {
    const l = LIMITE_DESPLAZAMIENTO;
    vistaObjetivo.centro.clampScalar(-l, l);
    vistaObjetivo.elevacion = THREE.MathUtils.clamp(vistaObjetivo.elevacion, ELEVACION_MIN, ELEVACION_MAX);
  }

  // Tamaño de la vista en el plano del centro, a la distancia actual de la cámara
  function anchoAltoVista(distancia = vistaActual.distancia) {
    const alto = 2 * distancia * Math.tan(THREE.MathUtils.degToRad(camara.fov / 2));
    return { ancho: alto * camara.aspect, alto };
  }

  const estaAfuera = (p, i) => objetivosPiezas[i] > 0 || p.libre.lengthSq() > 1e-8;

  function avisarPiezas() {
    alCambiarPiezas(piezas.map((p, i) => ({
      id: p.id, nombre: p.nombre, descripcion: p.descripcion, fija: p.fija,
      desmontada: estaAfuera(p, i), seleccionada: p.id === seleccionada, aislada: p.id === aislada,
    })));
  }

  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = contenedor;
    if (!w || !h) return;
    camara.aspect = w / h;
    camara.updateProjectionMatrix();
    renderer.setSize(w, h);
  }).observe(contenedor);

  // Centro de una pieza en coordenadas del contenedor del modelo, con su desplazamiento actual
  function centroActual(p) {
    const d = desplazamiento(p, p.actual);
    return p.centro.map((v, k) => v + d[k] + p.libreActual.getComponent(k) - centroModelo[k]);
  }

  renderer.setAnimationLoop(() => {
    const dt = reloj.getDelta();
    if (giro.x || giro.y) {
      objetivo.rotY += giro.x * dt;
      objetivo.rotX = THREE.MathUtils.clamp(objetivo.rotX - giro.y * dt, -Math.PI / 2, Math.PI / 2);
      const frenado = Math.exp(-dt / FRENADO_GIRO_S);
      giro.x *= frenado;
      giro.y *= frenado;
      if (Math.hypot(giro.x, giro.y) < 0.05) detenerGiro();
    }
    const a = 1 - Math.exp(-dt * RAPIDEZ_SUAVIZADO);
    vistaActual.centro.lerp(vistaObjetivo.centro, a);
    for (const k of ['distancia', 'azimut', 'elevacion']) vistaActual[k] += (vistaObjetivo[k] - vistaActual[k]) * a;
    ubicarCamara(camara, vistaActual);
    pivote.position.lerp(objetivo.pos, a);
    pivote.rotation.x += (objetivo.rotX - pivote.rotation.x) * a;
    pivote.rotation.y += (objetivo.rotY - pivote.rotation.y) * a;
    pivote.scale.setScalar(pivote.scale.x + (objetivo.escala - pivote.scale.x) * a);

    const ap = 1 - Math.exp(-dt * RAPIDEZ_PIEZAS);
    for (const p of piezas) {
      const i = piezas.indexOf(p);
      p.actual += (objetivosPiezas[i] - p.actual) * ap;
      p.libreActual.lerp(p.libre, a);
      p.obj.position.copy(p.base).add(new THREE.Vector3(...desplazamiento(p, p.actual))).add(p.libreActual);
      const opacidad = aislada && p.id !== aislada ? OPACIDAD_ATENUADA : 1;
      p.opacidad += (opacidad - p.opacidad) * ap;
      aplicarOpacidad(p);
    }
    if (contenedorModelo) {
      // Normal: el modelo entero, achicado si las piezas se alejan. Aislada: la pieza centrada y grande.
      const pieza = aislada && piezas.find((p) => p.id === aislada);
      const destino = pieza
        ? encuadreAislada(centroActual(pieza), pieza.radio, escalaBase, radioArmado)
        : { escala: escalaBase * ajusteEncuadre(), posicion: [0, 0, 0] };
      contenedorModelo.scale.setScalar(contenedorModelo.scale.x + (destino.escala - contenedorModelo.scale.x) * ap);
      contenedorModelo.position.lerp(new THREE.Vector3(...destino.posicion), ap);
    }
    renderer.render(escena, camara);
    alCuadro?.({ piezas: proyectarPiezas(), escala: objetivo.escala });
  });

  // Dónde se ve el centro de cada pieza, en coordenadas normalizadas de la vista (0,0 arriba a la izquierda)
  const proyectado = new THREE.Vector3();
  function proyectarPiezas() {
    if (!contenedorModelo) return [];
    return piezas.map((p) => {
      proyectado.fromArray(centroActual(p));
      contenedorModelo.localToWorld(proyectado).project(camara);
      return {
        id: p.id, nombre: p.nombre, descripcion: p.descripcion, color: p.color,
        x: (proyectado.x + 1) / 2, y: (1 - proyectado.y) / 2, opacidad: p.opacidad,
        seleccionada: p.id === seleccionada, aislada: p.id === aislada,
      };
    });
  }

  // Capas del estilo holograma para cada malla de una pieza: relleno tenue, aristas y nube de puntos.
  // Se crean una sola vez al cargar el modelo; cambiar de estilo solo las muestra u oculta.
  function prepararHolograma(raiz, color, areaModelo) {
    const mezcla = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
    const mallas = [];
    raiz.traverse((m) => { if (m.isMesh) mallas.push(m); });
    for (const m of mallas) {
      const relleno = new THREE.MeshBasicMaterial({ color, opacity: OPACIDAD_HOLO.relleno, side: THREE.DoubleSide, ...mezcla });
      relleno.userData = { holo: true, opacidadBase: OPACIDAD_HOLO.relleno };
      m.userData.materiales = { solido: m.material, holograma: relleno };

      const aristas = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 1),
        new THREE.LineBasicMaterial({ color, opacity: OPACIDAD_HOLO.aristas, ...mezcla }));
      aristas.material.userData = { holo: true, opacidadBase: OPACIDAD_HOLO.aristas };

      const pos = m.geometry.attributes.position.array;
      const indices = m.geometry.index?.array ?? null;
      const n = Math.round(Math.min(4000, Math.max(120, PUNTOS_POR_MODELO * areaSuperficie(pos, indices) / areaModelo)));
      const nube = new THREE.BufferGeometry();
      nube.setAttribute('position', new THREE.BufferAttribute(puntosEnSuperficie(pos, indices, n, crearAzar(n)), 3));
      const puntos = new THREE.Points(nube, new THREE.PointsMaterial({
        color, size: 0.03, sizeAttenuation: true, map: puntoRedondo, opacity: OPACIDAD_HOLO.puntos, ...mezcla,
      }));
      puntos.material.userData = { holo: true, opacidadBase: OPACIDAD_HOLO.puntos };

      for (const capa of [aristas, puntos]) {
        capa.userData.capaHolograma = true;
        capa.raycast = () => {}; // el cursor apunta a la malla, no a sus aristas ni a sus puntos
        m.add(capa);
      }
    }
    aplicarEstilo(raiz);
  }

  function aplicarEstilo(raiz) {
    raiz.traverse((o) => {
      if (o.userData.materiales) o.material = o.userData.materiales[estilo === 'holograma' ? 'holograma' : 'solido'];
      if (o.userData.capaHolograma) o.visible = estilo === 'holograma';
    });
  }

  function aplicarOpacidad(p) {
    const transparente = p.opacidad < 0.99;
    p.obj.traverse((m) => {
      // Holograma: la pieza elegida o apuntada se ve más llena
      if (m.material?.userData.holo) {
        m.material.opacity = Math.min(1, m.material.userData.opacidadBase * (1 + 6 * p.resalte)) * p.opacidad;
        return;
      }
      if (!m.isMesh) return;
      if (m.material.transparent !== transparente) {
        m.material.transparent = transparente;
        m.material.needsUpdate = true; // cambiar la transparencia exige recompilar el material
      }
      m.material.opacity = p.opacidad;
      m.material.depthWrite = p.opacidad > 0.5;
    });
  }

  // Radio que ocupa el objeto con las piezas en su posición actual, comparado con el objeto armado
  function ajusteEncuadre() {
    let radio = radioArmado;
    for (const p of piezas) radio = Math.max(radio, Math.hypot(...centroActual(p)) + p.radio);
    return radioArmado / radio;
  }

  // Lee las piezas del .glb. En los modelos propios, cada hijo de la escena con metadatos (extras → userData) es una
  // pieza; en los de Objaverse, las del nivel que elige nivelDePiezas, agrupadas por cercanía si son más de 80.
  // asociaciones (gltf.parser.associations) distingue los nodos del glTF de las primitivas de una malla.
  function elegirPiezas(raiz, asociaciones) {
    const esNodo = (o) => asociaciones?.get(o)?.nodes !== undefined;
    const esMalla = (o) => o.isMesh || asociaciones?.get(o)?.meshes !== undefined;
    const elegidas = nivelDePiezas(raiz, (o) => o.children.filter(esNodo), esMalla);
    // Pocas piezas y sin metadatos de desarme (un modelo que el autor no armó por piezas): se separa la malla
    const disenado = elegidas.some((o) => o.userData.pieza !== undefined || o.userData.orden !== undefined);
    if (elegidas.length < MINIMO_PIEZAS && !disenado) {
      const separadas = separarMallas(raiz);
      if (separadas) return separadas;
    }
    if (elegidas.length <= MAXIMO_PIEZAS) return elegidas;
    const centros = elegidas.map((o) => new THREE.Box3().setFromObject(o).getCenter(new THREE.Vector3()).toArray());
    return agruparPorCercania(centros).map((indices, k) => {
      if (indices.length === 1) return elegidas[indices[0]];
      const grupo = new THREE.Group();
      grupo.name = `Grupo ${k + 1} (${indices.length} partes)`;
      raiz.add(grupo);
      for (const i of indices) grupo.attach(elegidas[i]);
      return grupo;
    });
  }

  // Posiciones xyz planas (sirve también con atributos intercalados o cuantizados)
  function posicionesDe(geometria) {
    const a = geometria.attributes.position;
    const xyz = new Float32Array(a.count * 3);
    for (let i = 0; i < a.count; i++) {
      xyz[3 * i] = a.getX(i);
      xyz[3 * i + 1] = a.getY(i);
      xyz[3 * i + 2] = a.getZ(i);
    }
    return xyz;
  }

  // Una malla nueva con solo esos triángulos de la original (mismo material, mismo lugar). Los vértices se copian
  // compactos: así la caja de cada pieza es la suya y no la de la malla entera.
  function mallaParcial(malla, triangulos) {
    const g = malla.geometry;
    const indices = g.index?.array ?? null;
    const nuevo = new Int32Array(g.attributes.position.count).fill(-1);
    const viejos = [];
    const indice = new Uint32Array(triangulos.length * 3);
    let k = 0;
    for (const t of triangulos) {
      for (let c = 0; c < 3; c++) {
        const v = indices ? indices[3 * t + c] : 3 * t + c;
        if (nuevo[v] < 0) { nuevo[v] = viejos.length; viejos.push(v); }
        indice[k++] = nuevo[v];
      }
    }
    const geometria = new THREE.BufferGeometry();
    for (const [nombre, a] of Object.entries(g.attributes)) {
      const intercalado = a.isInterleavedBufferAttribute;
      const fuente = intercalado ? a.data.array : a.array;
      const paso = intercalado ? a.data.stride : a.itemSize;
      const desde = intercalado ? a.offset : 0;
      const datos = new fuente.constructor(viejos.length * a.itemSize);
      viejos.forEach((v, i) => {
        for (let j = 0; j < a.itemSize; j++) datos[i * a.itemSize + j] = fuente[v * paso + desde + j];
      });
      geometria.setAttribute(nombre, new THREE.BufferAttribute(datos, a.itemSize, a.normalized));
    }
    geometria.setIndex(new THREE.BufferAttribute(viejos.length < 65536 ? Uint16Array.from(indice) : indice, 1));
    const parte = new THREE.Mesh(geometria, malla.material);
    parte.position.copy(malla.position);
    parte.quaternion.copy(malla.quaternion);
    parte.scale.copy(malla.scale);
    malla.parent.add(parte);
    return parte;
  }

  // Separa las mallas del modelo en partes sueltas y, si son pocas, corta la más grande en secciones
  // (componentesConexas y seccionar de piezas.js). Las piezas resultantes reemplazan a las mallas originales.
  // Devuelve las piezas nuevas, o null si no se pudo separar en más de una.
  // Los modelos animados (huesos o deformaciones) se congelan en su pose: una malla fija con los vértices donde
  // los deja el esqueleto. El visor no reproduce animaciones, así que se ve igual, y ya se puede cortar.
  function congelarPose(malla) {
    const geometria = malla.geometry.clone();
    geometria.morphAttributes = {};
    if (malla.isSkinnedMesh) {
      const p = geometria.attributes.position;
      const xyz = new Float32Array(p.count * 3);
      const v = new THREE.Vector3();
      for (let i = 0; i < p.count; i++) {
        malla.applyBoneTransform(i, v.fromBufferAttribute(p, i));   // queda en coordenadas de la malla
        xyz.set([v.x, v.y, v.z], 3 * i);
      }
      geometria.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
      geometria.deleteAttribute('skinIndex');
      geometria.deleteAttribute('skinWeight');
      geometria.computeVertexNormals();
    }
    const fija = new THREE.Mesh(geometria, malla.material);
    fija.name = malla.name;
    fija.position.copy(malla.position);
    fija.quaternion.copy(malla.quaternion);
    fija.scale.copy(malla.scale);
    malla.parent.add(fija);
    malla.removeFromParent();
    return fija;
  }

  function separarMallas(raiz) {
    const originales = [];
    raiz.traverse((o) => { if (o.isMesh && !Array.isArray(o.material)) originales.push(o); });
    const mallas = originales.map((o) => (o.isSkinnedMesh || o.geometry.morphAttributes?.position ? congelarPose(o) : o));
    let partes = [];   // { malla, triangulos, tipo: 'Pieza' (suelta) | 'Sección' (corte), centro (en el mundo) }
    const datos = new Map();
    for (const malla of mallas) {
      const xyz = posicionesDe(malla.geometry);
      const indices = malla.geometry.index?.array ?? null;
      datos.set(malla, { xyz, indices });
      for (const triangulos of componentesConexas(xyz, indices)) partes.push({ malla, triangulos, tipo: 'Pieza' });
    }
    if (partes.length && hayQueSeccionar(partes.map((p) => p.triangulos.length))) {
      const mayor = partes.reduce((a, b) => (b.triangulos.length > a.triangulos.length ? b : a));
      const { xyz, indices } = datos.get(mayor.malla);
      const secciones = seccionar(xyz, indices, mayor.triangulos, SECCIONES)
        .map((triangulos) => ({ malla: mayor.malla, triangulos, tipo: 'Sección' }));
      partes = [...partes.filter((p) => p !== mayor), ...secciones];
      separacion = 'secciones';
    } else {
      separacion = 'partes';
    }
    if (partes.length < 2) {
      separacion = null;
      return null;
    }
    for (const p of partes) {
      const { xyz, indices } = datos.get(p.malla);
      p.centro = p.malla.localToWorld(new THREE.Vector3(...centroTriangulos(xyz, indices, p.triangulos))).toArray();
    }
    // Más de 80 partes sueltas (tornillos, hojas, letras): se juntan por cercanía antes de crear las mallas
    const grupos = agruparPorCercania(partes.map((p) => p.centro));
    const caja = new THREE.Box3().setFromObject(raiz);
    const centroModelo = caja.getCenter(new THREE.Vector3()).toArray();
    const centros = grupos.map((g) => [0, 1, 2].map((k) => g.reduce((s, i) => s + partes[i].centro[k], 0) / g.length));
    const nombres = nombresPorPosicion(centros, centroModelo, caja.getSize(new THREE.Vector3()).length() || 1,
      grupos.map((g) => partes[g[0]].tipo));
    const piezasNuevas = grupos.map((g, j) => {
      // Una malla parcial por cada malla original que aporta triángulos a este grupo
      const porMalla = new Map();
      for (const i of g) {
        const { malla, triangulos } = partes[i];
        if (!porMalla.has(malla)) porMalla.set(malla, []);
        porMalla.get(malla).push(triangulos);
      }
      // (flat y no push(...): una parte puede tener millones de triángulos y desbordar la pila)
      const trozos = [...porMalla].map(([malla, listas]) => mallaParcial(malla, listas.flat()));
      let pieza = trozos[0];
      if (trozos.length > 1) {
        pieza = new THREE.Group();
        raiz.add(pieza);
        for (const t of trozos) pieza.attach(t);
      }
      pieza.name = nombres[j];
      return pieza;
    });
    for (const malla of mallas) malla.removeFromParent();
    return piezasNuevas;
  }

  function leerPiezas(raiz, caja, asociaciones) {
    const centro = caja.getCenter(new THREE.Vector3()).toArray();
    const tamano = caja.getSize(new THREE.Vector3()).length() || 1;
    const elegidas = elegirPiezas(raiz, asociaciones);
    // El desplazamiento de desmontaje está en coordenadas del modelo: las piezas que estaban más abajo en el árbol
    // pasan a colgar de la escena, sin moverse en el mundo
    for (const obj of elegidas) if (obj.parent !== raiz) raiz.attach(obj);
    const ids = new Set();
    const datos = elegidas.map((obj, i) => {
      let id = obj.userData.pieza ?? (obj.name || `pieza_${i + 1}`);
      while (ids.has(id)) id = `${id}_${i + 1}`;
      ids.add(id);
      const cajaPieza = new THREE.Box3().setFromObject(obj);
      const t = cajaPieza.getSize(new THREE.Vector3());
      // Cada pieza con materiales propios, para poder resaltarla sin afectar a las demás
      const materiales = new Set();
      obj.traverse((m) => {
        if (!m.isMesh) return;
        m.material = m.material.clone();
        const nombre = nombreMaterial(m.material.name);
        if (nombre) materiales.add(nombre);
      });
      return {
        obj,
        id,
        nombre: obj.userData.nombre ?? (obj.name || `Pieza ${i + 1}`),
        descripcion: obj.userData.descripcion ?? '',
        orden: obj.userData.orden,
        direccion: obj.userData.direccion,
        centro: cajaPieza.getCenter(new THREE.Vector3()).toArray(),
        tamano: t.toArray(),
        materiales: [...materiales],
        volumen: t.x * t.y * t.z,
        radio: t.length() / 2,
        base: obj.position.clone(),
        actual: 0,
        libre: new THREE.Vector3(),        // desplazamiento a mano (pieza tomada con un pellizco)
        libreActual: new THREE.Vector3(),
        opacidad: 1,
        resalte: 0,
      };
    });
    return datos.length > 1 ? prepararPiezas(datos, centro, tamano) : [];
  }

  // Si una pieza sale hacia la cámara (eje Z) y la vista está de frente, el despiece no se ve:
  // se gira a una vista tres cuartos, un poco desde arriba. Si el usuario ya giró el objeto, se respeta.
  function verDespiece(indices) {
    if (Math.abs(objetivo.rotY) > 0.15 || Math.abs(objetivo.rotX) > 0.15) return;
    if (indices.some((i) => Math.abs(piezas[i].dir[2]) > 0.7)) {
      objetivo.rotY = -0.7;
      objetivo.rotX = 0.3;
    }
  }

  function resaltar() {
    for (const p of piezas) {
      const intensidad = p.id === seleccionada ? 0.35 : p.id === apuntada ? 0.18 : 0;
      p.resalte = intensidad;
      p.obj.traverse((m) => {
        const material = m.userData.materiales?.solido ?? m.material;
        if (!m.isMesh || !material.emissive) return;
        material.emissive.copy(intensidad ? COLOR_SELECCION : NEGRO);
        material.emissiveIntensity = intensidad;
      });
    }
  }

  // Pieza bajo un punto en coordenadas normalizadas de la vista (0,0 arriba a la izquierda), o null.
  // Mientras se analiza una pieza aislada, las demás (casi invisibles) no se pueden apuntar.
  function piezaEnVista(u, v) {
    raycaster.setFromCamera(new THREE.Vector2(u * 2 - 1, -(v * 2 - 1)), camara);
    const candidatas = piezas.filter((p) => !aislada || p.id === aislada);
    const [impacto] = raycaster.intersectObjects(candidatas.map((p) => p.obj), true);
    if (!impacto) return null;
    return candidatas.find((p) => {
      let o = impacto.object;
      while (o && o !== p.obj) o = o.parent;
      return o === p.obj;
    })?.id ?? null;
  }

  return {
    async cargarModelo(url) {
      const gltf = await loader.loadAsync(url);
      modelo = gltf.scene;
      unidad = gltf.scene.userData.unidad ?? null;
      // Centrar y llevar a un tamaño comparable, sin importar las unidades del archivo
      modelo.updateMatrixWorld(true);
      const caja = new THREE.Box3().setFromObject(modelo);
      separacion = null;
      piezas = leerPiezas(modelo, caja, gltf.parser.associations);
      objetivosPiezas = piezas.map(() => 0);
      // Estilo holograma: un color por pieza (o uno solo si el modelo no está armado por piezas)
      let areaModelo = 0;
      modelo.traverse((m) => {
        if (m.isMesh) areaModelo += areaSuperficie(m.geometry.attributes.position.array, m.geometry.index?.array ?? null);
      });
      piezas.forEach((p, i) => { p.color = colorPieza(i); });
      if (piezas.length) piezas.forEach((p) => prepararHolograma(p.obj, p.color, areaModelo || 1));
      else prepararHolograma(modelo, colorPieza(0), areaModelo || 1);
      seleccionada = apuntada = aislada = null;
      const tam = caja.getSize(new THREE.Vector3()).length() || 1;
      modelo.position.sub(caja.getCenter(new THREE.Vector3()));
      contenedorModelo = new THREE.Group();
      contenedorModelo.add(modelo);
      escalaBase = 2.6 / tam;
      radioArmado = tam / 2;
      centroModelo = caja.getCenter(new THREE.Vector3()).toArray();
      contenedorModelo.scale.setScalar(escalaBase);
      pivote.clear();
      pivote.add(contenedorModelo);
      historial.limpiar(); // deshacer no cruza de un modelo a otro
      this.reiniciar();
    },

    // Guarda el estado actual para poder volver con deshacer(). Se llama antes de cada cambio.
    guardarEstado() {
      historial.guardar(estadoActual());
    },

    // Vuelve al estado anterior (vista y piezas). Devuelve false si no había nada para deshacer.
    deshacer() {
      const anterior = historial.deshacer(estadoActual());
      if (!anterior) return false;
      detenerGiro();
      objetivo.pos.fromArray(anterior.pos);
      ({ rotX: objetivo.rotX, rotY: objetivo.rotY, escala: objetivo.escala } = anterior);
      if (anterior.piezas?.length === objetivosPiezas.length) objetivosPiezas = [...anterior.piezas];
      if (anterior.libres?.length === piezas.length) piezas.forEach((p, i) => p.libre.fromArray(anterior.libres[i]));
      aislada = anterior.aislada ?? null;
      avisarPiezas();
      return true;
    },

    // dx, dy en fracción de pantalla (positivo = derecha / arriba)
    desplazar(dx, dy) {
      // Se mueve en el plano de la pantalla (la cámara puede estar girada) y no sale de la vista
      const { ancho, alto } = anchoAltoVista();
      const { derecha, arriba } = ejesPantalla();
      objetivo.pos.addScaledVector(derecha, dx * ancho).addScaledVector(arriba, dy * alto);
      const relativa = objetivo.pos.clone().sub(vistaObjetivo.centro);
      const x = relativa.dot(derecha);
      const y = relativa.dot(arriba);
      objetivo.pos
        .addScaledVector(derecha, THREE.MathUtils.clamp(x, -ancho / 2, ancho / 2) - x)
        .addScaledVector(arriba, THREE.MathUtils.clamp(y, -alto / 2, alto / 2) - y);
    },

    // ------------------------------------------------- navegación con el mouse

    /**
     * Acerca (factor < 1) o aleja (factor > 1) la cámara hacia el punto de la pantalla bajo el mouse:
     * ese punto queda quieto, como en un mapa. xCliente, yCliente en píxeles de la ventana.
     */
    acercarCamara(factor, xCliente, yCliente) {
      const d = vistaObjetivo.distancia;
      const nueva = THREE.MathUtils.clamp(d * factor, DISTANCIA_MIN, DISTANCIA_MAX);
      if (nueva === d) return;
      const guia = guiaEnObjetivo();
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector3(((xCliente - r.left) / r.width) * 2 - 1, -((yCliente - r.top) / r.height) * 2 + 1, 0.5);
      const rayo = ndc.unproject(guia).sub(guia.position).normalize();
      const frente = vistaObjetivo.centro.clone().sub(guia.position).normalize();
      // Punto bajo el mouse en el plano del centro (perpendicular a la mirada): todo se acerca hacia él
      const punto = guia.position.clone().addScaledVector(rayo, d / rayo.dot(frente));
      vistaObjetivo.centro.sub(punto).multiplyScalar(nueva / d).add(punto);
      vistaObjetivo.distancia = nueva;
      limitarVista();
    },

    /** Desplaza la vista siguiendo un arrastre del mouse (en píxeles): lo que está abajo del mouse lo acompaña. */
    arrastrarCamara(dxPx, dyPx) {
      const { alto } = anchoAltoVista(vistaObjetivo.distancia);
      const porPixel = alto / renderer.domElement.clientHeight;
      const { derecha, arriba } = ejesPantalla(guiaEnObjetivo());
      vistaObjetivo.centro.addScaledVector(derecha, -dxPx * porPixel).addScaledVector(arriba, dyPx * porPixel);
      limitarVista();
    },

    /**
     * Gira la cámara alrededor del centro de la vista siguiendo un arrastre (en píxeles):
     * a los costados da la vuelta alrededor del objeto; arriba y abajo lo muestra desde más arriba o más abajo.
     * Cruzar todo el ancho de la pantalla = una vuelta.
     */
    orbitarCamara(dxPx, dyPx) {
      const radPorPixel = (2 * Math.PI) / renderer.domElement.clientWidth;
      vistaObjetivo.azimut -= dxPx * radPorPixel;
      vistaObjetivo.elevacion += dyPx * radPorPixel;
      limitarVista();
    },

    // dx, dy en radianes (positivo = el frente del objeto va a la derecha / arriba)
    rotar(dx, dy) {
      detenerGiro();
      objetivo.rotY += dx;
      objetivo.rotX = THREE.MathUtils.clamp(objetivo.rotX - dy, -Math.PI / 2, Math.PI / 2);
    },

    // Vistas predefinidas del pedido por voz/texto (parametros.vista)
    vista(nombre) {
      const VISTAS = {
        frontal: [0, 0],
        trasera: [0, Math.PI],
        lateral: [0, Math.PI / 2],
        superior: [Math.PI / 2, 0],
      };
      if (!VISTAS[nombre]) return;
      detenerGiro();
      [objetivo.rotX, objetivo.rotY] = VISTAS[nombre];
    },

    /** Deja el objeto girando a vx, vy rad/s (mismos signos que rotar), frenando solo. */
    impulsar(vx, vy) {
      giro.x = vx;
      giro.y = vy;
    },
    detenerGiro,
    girando: () => Boolean(giro.x || giro.y),

    escala: () => objetivo.escala,

    escalarA(factor) {
      objetivo.escala = THREE.MathUtils.clamp(factor, ESCALA_MIN, ESCALA_MAX);
    },

    // Vuelve a la vista inicial con el objeto armado
    reiniciar() {
      detenerGiro();
      Object.assign(vistaObjetivo, copiarVista(vistaInicial));
      objetivo.pos.set(0, 0, 0);
      objetivo.rotX = 0;
      objetivo.rotY = 0;
      objetivo.escala = 1;
      aislada = null;
      this.armarTodo();
    },

    // ---------------------------------------------------------------- piezas

    tienePiezas: () => piezas.length > 1,
    separacion: () => separacion,

    hayPiezasAfuera: () => piezas.some(estaAfuera),

    /** Saca la próxima pieza en orden de desmontaje. Devuelve su nombre, o null si no queda ninguna. */
    desmontarSiguiente() {
      const i = siguienteParaDesmontar(piezas, objetivosPiezas);
      if (i < 0) return null;
      verDespiece([i]);
      objetivosPiezas[i] = 1;
      seleccionada = piezas[i].id;
      resaltar();
      avisarPiezas();
      return piezas[i].nombre;
    },

    /** Vuelve a montar la última pieza desmontada. Devuelve su nombre, o null. */
    montarUltima() {
      const i = ultimaDesmontada(piezas, objetivosPiezas);
      if (i < 0) return null;
      objetivosPiezas[i] = 0;
      piezas[i].libre.set(0, 0, 0);
      avisarPiezas();
      return piezas[i].nombre;
    },

    armarTodo() {
      objetivosPiezas = piezas.map(() => 0);
      for (const p of piezas) p.libre.set(0, 0, 0);
      avisarPiezas();
    },

    /** Vista explotada: todas las piezas desmontables se alejan una fracción (0 a 1). */
    explotar(fraccion = 1) {
      if (fraccion > 0) verDespiece(piezas.map((p, i) => i).filter((i) => !piezas[i].fija));
      objetivosPiezas = piezas.map((p) => (p.fija ? 0 : fraccion));
      avisarPiezas();
    },

    seleccionar(id) {
      seleccionada = id;
      resaltar();
      avisarPiezas();
    },

    seleccionada: () => seleccionada,

    /** Busca una pieza por nombre o id, sin tildes ni mayúsculas ("el volante" → "volante"). */
    buscarPieza(texto) {
      const n = (t) => t.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
      const q = n(texto);
      return piezas.find((p) => n(p.nombre) === q || n(p.id) === q)?.id
        ?? piezas.find((p) => q.includes(n(p.nombre)) || n(p.nombre).includes(q))?.id ?? null;
    },

    /** Pieza bajo un punto de la pantalla (para seleccionarla con un clic), o null. */
    piezaEn(xCliente, yCliente) {
      const r = renderer.domElement.getBoundingClientRect();
      return piezaEnVista((xCliente - r.left) / r.width, (yCliente - r.top) / r.height);
    },

    /**
     * Cursor de la mano en coordenadas normalizadas de la vista (null = ocultarlo). Resalta la pieza
     * apuntada y la devuelve. progreso (0-1) dibuja el aro de la selección por permanencia.
     */
    apuntar(u, v, progreso = 0) {
      if (u == null) {
        cursor.hidden = true;
        if (apuntada) {
          apuntada = null;
          resaltar();
        }
        return null;
      }
      cursor.hidden = false;
      cursor.style.left = `${u * 100}%`;
      cursor.style.top = `${v * 100}%`;
      cursor.style.setProperty('--progreso', `${Math.round(progreso * 360)}deg`);
      const id = piezas.length > 1 ? piezaEnVista(u, v) : null;
      if (id !== apuntada) {
        apuntada = id;
        resaltar();
      }
      return id;
    },

    /** Mueve solo una pieza (tomada con un pellizco). dx, dy en fracción de pantalla. */
    moverPieza(id, dx, dy) {
      const p = piezas.find((q) => q.id === id);
      if (!p || !modelo) return;
      const { ancho, alto } = anchoAltoVista();
      // El desplazamiento en pantalla se lleva a las coordenadas del modelo (que puede estar girado y escalado)
      modelo.updateMatrixWorld(true);
      const origen = modelo.localToWorld(p.obj.position.clone());
      const { derecha, arriba } = ejesPantalla();
      const destino = origen.clone().addScaledVector(derecha, dx * ancho).addScaledVector(arriba, dy * alto);
      p.libre.add(modelo.worldToLocal(destino).sub(modelo.worldToLocal(origen)));
      avisarPiezas();
    },

    /** Deja una pieza sola, centrada y grande, para analizarla. Devuelve su información, o null. */
    aislar(id) {
      if (!piezas.some((p) => p.id === id)) return null;
      aislada = id;
      seleccionada = id;
      resaltar();
      avisarPiezas();
      return this.infoPieza(id);
    },

    salirDeAislamiento() {
      if (!aislada) return false;
      aislada = null;
      avisarPiezas();
      return true;
    },

    aislada: () => aislada,

    /** Lo que se sabe de una pieza para analizarla: nombre, función, materiales y medidas. */
    infoPieza(id) {
      const p = piezas.find((q) => q.id === id);
      if (!p) return null;
      return {
        id: p.id, nombre: p.nombre, descripcion: p.descripcion, fija: p.fija,
        materiales: p.materiales, medidas: textoMedidas(p.tamano, unidad),
        orden: p.fija ? null : p.orden,
        reconocida: p.reconocida ?? null,
      };
    },

    /**
     * Pone nombre a las piezas de un motor que no lo traen ("Sección superior", "Object_12"), según su forma
     * (reconocer_piezas.js). red: la de cargarRed(). familia: 'combustion', 'electrico' o 'motor'.
     * Las piezas con un nombre puesto por el autor del modelo, o de las que la red no está segura, quedan como
     * están. Devuelve { nombradas, total }, o null si mientras tanto se cargó otro modelo.
     */
    async reconocerPiezas(red, familia = 'motor') {
      const actual = modelo;
      if (!actual || piezas.length < 2) return { nombradas: 0, total: piezas.length };
      const renombrables = piezas.map((p) => nombreGenerico(p.nombre));
      if (!renombrables.some(Boolean)) return { nombradas: 0, total: piezas.length };
      // Cada malla en coordenadas del modelo, con las piezas armadas (aunque ya se haya empezado a desarmar)
      const corridas = piezas.map((p) => p.obj.position.clone());
      piezas.forEach((p) => p.obj.position.copy(p.base));
      actual.updateMatrixWorld(true);
      const inversa = actual.matrixWorld.clone().invert();
      const relativa = new THREE.Matrix4();
      const fuentes = piezas.map((p) => {
        const lista = [];
        p.obj.traverse((m) => {
          if (!m.isMesh) return;
          relativa.multiplyMatrices(inversa, m.matrixWorld);
          lista.push({ xyz: posicionesDe(m.geometry), indices: m.geometry.index?.array ?? null, matriz: relativa.toArray() });
        });
        return lista;
      });
      piezas.forEach((p, i) => p.obj.position.copy(corridas[i]));
      actual.updateMatrixWorld(true);

      const nombres = await reconocer(red, fuentes, piezas.map((p) => p.centro),
        { familia, azar: crearAzar(1), renombrables });
      if (modelo !== actual) return null;
      let nombradas = 0;
      piezas.forEach((p, i) => {
        const n = nombres[i];
        if (!n) return;
        p.reconocida = { clase: n.clase, confianza: n.confianza, nombreOriginal: p.nombre };
        p.nombre = n.nombre;
        p.descripcion = n.descripcion;
        nombradas++;
      });
      avisarPiezas();
      return { nombradas, total: piezas.length };
    },

    /** 'solido' (materiales del modelo) o 'holograma' (piezas de colores con aristas y puntos). */
    fijarEstilo(nuevo) {
      estilo = nuevo;
      if (modelo) aplicarEstilo(modelo);
    },

    estilo: () => estilo,

    /** La grilla del piso: se oculta cuando el video de la cámara es el fondo. */
    mostrarGrilla(visible) {
      grilla.visible = visible;
    },

    elemento: renderer.domElement,
  };
}
