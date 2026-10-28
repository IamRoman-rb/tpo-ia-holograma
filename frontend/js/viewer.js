// Escena Three.js: carga de .glb, transformaciones suavizadas del objeto y desmontaje por piezas.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { crearHistorial } from './historial.js';
import { desplazamiento, prepararPiezas, siguienteParaDesmontar, ultimaDesmontada } from './piezas.js';

export const ESCALA_MIN = 0.2;   // mismos límites que backend/safety.py
export const ESCALA_MAX = 5;
const DISTANCIA_CAMARA = 4;
const RAPIDEZ_SUAVIZADO = 14;   // más alto = sigue a la mano más rápido
const RAPIDEZ_PIEZAS = 6;       // el desmontaje es más lento, para que se vea el recorrido
const COLOR_SELECCION = new THREE.Color(0x2bd4ff);

/** alCambiarPiezas(lista) se llama cada vez que cambia el estado de las piezas. */
export function crearVisor(contenedor, { alCambiarPiezas = () => {} } = {}) {
  const escena = new THREE.Scene();
  const camara = new THREE.PerspectiveCamera(50, contenedor.clientWidth / contenedor.clientHeight, 0.1, 100);
  camara.position.set(0, 0.3, DISTANCIA_CAMARA);
  camara.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(contenedor.clientWidth, contenedor.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  contenedor.appendChild(renderer.domElement);

  // Iluminación: entorno para los reflejos metálicos + contraluz cian de "holograma"
  const pmrem = new THREE.PMREMGenerator(renderer);
  escena.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const contraluz = new THREE.DirectionalLight(0x5ee7ff, 2.5);
  contraluz.position.set(-3, 2, -4);
  escena.add(contraluz);

  const grilla = new THREE.GridHelper(8, 32, 0x2bd4ff, 0x16475a);
  grilla.position.y = -1.3;
  grilla.material.transparent = true;
  grilla.material.opacity = 0.35;
  escena.add(grilla);

  // El pivote es lo que se manipula; el modelo cargado vive adentro, centrado.
  const pivote = new THREE.Group();
  escena.add(pivote);

  // Estado objetivo: los gestos lo modifican y el pivote lo sigue con suavizado.
  const objetivo = { pos: new THREE.Vector3(), rotX: 0, rotY: 0, escala: 1 };

  // Piezas del modelo actual: { id, nombre, descripcion, fija, dir, dist, obj, base, actual }
  let piezas = [];
  let objetivosPiezas = [];      // 0 = montada · 1 = desmontada (puede ser intermedio: vista explotada)
  let seleccionada = null;
  // Encuadre: al desarmar, el modelo se achica para que todas las piezas sigan a la vista
  let contenedorModelo = null;
  let escalaBase = 1;
  let radioArmado = 1;
  let centroModelo = [0, 0, 0];

  const historial = crearHistorial();
  const estadoActual = () => ({
    pos: objetivo.pos.toArray().map((v) => +v.toFixed(4)),
    rotX: +objetivo.rotX.toFixed(4), rotY: +objetivo.rotY.toFixed(4), escala: +objetivo.escala.toFixed(4),
    piezas: objetivosPiezas.map((v) => +v.toFixed(3)),
  });
  const loader = new GLTFLoader();
  const reloj = new THREE.Clock();
  const raycaster = new THREE.Raycaster();

  function anchoAltoVista() {
    const alto = 2 * DISTANCIA_CAMARA * Math.tan(THREE.MathUtils.degToRad(camara.fov / 2));
    return { ancho: alto * camara.aspect, alto };
  }

  function avisarPiezas() {
    alCambiarPiezas(piezas.map((p, i) => ({
      id: p.id, nombre: p.nombre, descripcion: p.descripcion, fija: p.fija,
      desmontada: objetivosPiezas[i] > 0, seleccionada: p.id === seleccionada,
    })));
  }

  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = contenedor;
    if (!w || !h) return;
    camara.aspect = w / h;
    camara.updateProjectionMatrix();
    renderer.setSize(w, h);
  }).observe(contenedor);

  renderer.setAnimationLoop(() => {
    const dt = reloj.getDelta();
    const a = 1 - Math.exp(-dt * RAPIDEZ_SUAVIZADO);
    pivote.position.lerp(objetivo.pos, a);
    pivote.rotation.x += (objetivo.rotX - pivote.rotation.x) * a;
    pivote.rotation.y += (objetivo.rotY - pivote.rotation.y) * a;
    pivote.scale.setScalar(pivote.scale.x + (objetivo.escala - pivote.scale.x) * a);

    const ap = 1 - Math.exp(-dt * RAPIDEZ_PIEZAS);
    piezas.forEach((p, i) => {
      p.actual += (objetivosPiezas[i] - p.actual) * ap;
      p.obj.position.copy(p.base).add(new THREE.Vector3(...desplazamiento(p, p.actual)));
    });
    if (contenedorModelo) contenedorModelo.scale.setScalar(escalaBase * ajusteEncuadre());
    renderer.render(escena, camara);
  });

  // Radio que ocupa el objeto con las piezas en su posición actual, comparado con el objeto armado
  function ajusteEncuadre() {
    let radio = radioArmado;
    for (const p of piezas) {
      const d = desplazamiento(p, p.actual);
      const centro = p.centro.map((v, k) => v + d[k] - centroModelo[k]);
      radio = Math.max(radio, Math.hypot(...centro) + p.radio);
    }
    return radioArmado / radio;
  }

  // Lee las piezas del .glb: cada hijo de la escena con metadatos (extras → userData) es una pieza
  function leerPiezas(modelo, caja) {
    const centroModelo = caja.getCenter(new THREE.Vector3()).toArray();
    const tamano = caja.getSize(new THREE.Vector3()).length() || 1;
    const datos = modelo.children.map((obj, i) => {
      const cajaPieza = new THREE.Box3().setFromObject(obj);
      const t = cajaPieza.getSize(new THREE.Vector3());
      // Cada pieza con materiales propios, para poder resaltarla sin afectar a las demás
      obj.traverse((m) => { if (m.isMesh) m.material = m.material.clone(); });
      return {
        obj,
        id: obj.userData.pieza ?? obj.name ?? `pieza_${i + 1}`,
        nombre: obj.userData.nombre ?? obj.name ?? `Pieza ${i + 1}`,
        descripcion: obj.userData.descripcion ?? '',
        orden: obj.userData.orden,
        direccion: obj.userData.direccion,
        centro: cajaPieza.getCenter(new THREE.Vector3()).toArray(),
        volumen: t.x * t.y * t.z,
        radio: t.length() / 2,
        base: obj.position.clone(),
        actual: 0,
      };
    });
    return datos.length > 1 ? prepararPiezas(datos, centroModelo, tamano) : [];
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
      p.obj.traverse((m) => {
        if (!m.isMesh || !m.material.emissive) return;
        m.material.emissive.copy(p.id === seleccionada ? COLOR_SELECCION : new THREE.Color(0x000000));
        m.material.emissiveIntensity = p.id === seleccionada ? 0.35 : 0;
      });
    }
  }

  return {
    async cargarModelo(url) {
      const gltf = await loader.loadAsync(url);
      const modelo = gltf.scene;
      // Centrar y llevar a un tamaño comparable, sin importar las unidades del archivo
      modelo.updateMatrixWorld(true);
      const caja = new THREE.Box3().setFromObject(modelo);
      piezas = leerPiezas(modelo, caja);
      objetivosPiezas = piezas.map(() => 0);
      seleccionada = null;
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
      objetivo.pos.fromArray(anterior.pos);
      ({ rotX: objetivo.rotX, rotY: objetivo.rotY, escala: objetivo.escala } = anterior);
      if (anterior.piezas?.length === objetivosPiezas.length) objetivosPiezas = [...anterior.piezas];
      avisarPiezas();
      return true;
    },

    // dx, dy en fracción de pantalla (positivo = derecha / arriba)
    desplazar(dx, dy) {
      const { ancho, alto } = anchoAltoVista();
      objetivo.pos.x = THREE.MathUtils.clamp(objetivo.pos.x + dx * ancho, -ancho / 2, ancho / 2);
      objetivo.pos.y = THREE.MathUtils.clamp(objetivo.pos.y + dy * alto, -alto / 2, alto / 2);
    },

    // dx, dy en radianes (positivo = el frente del objeto va a la derecha / arriba)
    rotar(dx, dy) {
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
      [objetivo.rotX, objetivo.rotY] = VISTAS[nombre];
    },

    escala: () => objetivo.escala,

    escalarA(factor) {
      objetivo.escala = THREE.MathUtils.clamp(factor, ESCALA_MIN, ESCALA_MAX);
    },

    // Vuelve a la vista inicial con el objeto armado
    reiniciar() {
      objetivo.pos.set(0, 0, 0);
      objetivo.rotX = 0;
      objetivo.rotY = 0;
      objetivo.escala = 1;
      this.armarTodo();
    },

    // ---------------------------------------------------------------- piezas

    tienePiezas: () => piezas.length > 1,

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
      avisarPiezas();
      return piezas[i].nombre;
    },

    armarTodo() {
      objetivosPiezas = piezas.map(() => 0);
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

    /** Pieza bajo un punto de la pantalla (para seleccionarla con un clic), o null. */
    piezaEn(xCliente, yCliente) {
      const r = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(((xCliente - r.left) / r.width) * 2 - 1,
        -((yCliente - r.top) / r.height) * 2 + 1), camara);
      const [impacto] = raycaster.intersectObjects(piezas.map((p) => p.obj), true);
      if (!impacto) return null;
      return piezas.find((p) => {
        let o = impacto.object;
        while (o && o !== p.obj) o = o.parent;
        return o === p.obj;
      })?.id ?? null;
    },

    elemento: renderer.domElement,
  };
}
