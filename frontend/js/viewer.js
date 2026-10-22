// Escena Three.js: carga de .glb y transformaciones suavizadas del objeto.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export const ESCALA_MIN = 0.2;   // mismos límites que backend/safety.py
export const ESCALA_MAX = 5;
const DISTANCIA_CAMARA = 4;
const RAPIDEZ_SUAVIZADO = 14;   // más alto = sigue a la mano más rápido

export function crearVisor(contenedor) {
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
  const loader = new GLTFLoader();
  const reloj = new THREE.Clock();

  function anchoAltoVista() {
    const alto = 2 * DISTANCIA_CAMARA * Math.tan(THREE.MathUtils.degToRad(camara.fov / 2));
    return { ancho: alto * camara.aspect, alto };
  }

  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = contenedor;
    if (!w || !h) return;
    camara.aspect = w / h;
    camara.updateProjectionMatrix();
    renderer.setSize(w, h);
  }).observe(contenedor);

  renderer.setAnimationLoop(() => {
    const a = 1 - Math.exp(-reloj.getDelta() * RAPIDEZ_SUAVIZADO);
    pivote.position.lerp(objetivo.pos, a);
    pivote.rotation.x += (objetivo.rotX - pivote.rotation.x) * a;
    pivote.rotation.y += (objetivo.rotY - pivote.rotation.y) * a;
    pivote.scale.setScalar(pivote.scale.x + (objetivo.escala - pivote.scale.x) * a);
    renderer.render(escena, camara);
  });

  return {
    async cargarModelo(url) {
      const gltf = await loader.loadAsync(url);
      const modelo = gltf.scene;
      // Centrar y llevar a un tamaño comparable, sin importar las unidades del archivo
      const caja = new THREE.Box3().setFromObject(modelo);
      const tam = caja.getSize(new THREE.Vector3()).length() || 1;
      modelo.position.sub(caja.getCenter(new THREE.Vector3()));
      const contenedorModelo = new THREE.Group();
      contenedorModelo.add(modelo);
      contenedorModelo.scale.setScalar(2.6 / tam);
      pivote.clear();
      pivote.add(contenedorModelo);
      this.reiniciar();
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

    reiniciar() {
      objetivo.pos.set(0, 0, 0);
      objetivo.rotX = 0;
      objetivo.rotY = 0;
      objetivo.escala = 1;
    },
  };
}
