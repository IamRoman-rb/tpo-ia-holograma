// Escena Three.js: carga de .glb y manipulación con gestos.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export function crearVisor(contenedor) {
  const escena = new THREE.Scene();
  const camara = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
  camara.position.set(0, 0, 4);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(devicePixelRatio);
  contenedor.appendChild(renderer.domElement);

  escena.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2));
  const luz = new THREE.DirectionalLight(0xffffff, 2);
  luz.position.set(3, 5, 4);
  escena.add(luz);

  // Grupo que se manipula; el modelo cargado vive adentro.
  const pivote = new THREE.Group();
  escena.add(pivote);
  // Placeholder hasta que haya modelos en el catálogo
  pivote.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x4f8cff })));

  const loader = new GLTFLoader();

  addEventListener('resize', () => {
    camara.aspect = innerWidth / innerHeight;
    camara.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  renderer.setAnimationLoop(() => renderer.render(escena, camara));

  return {
    async cargarModelo(url) {
      const gltf = await loader.loadAsync(url);
      pivote.clear();
      // Centrar y llevar a un tamaño comparable
      const caja = new THREE.Box3().setFromObject(gltf.scene);
      const tam = caja.getSize(new THREE.Vector3()).length();
      gltf.scene.position.sub(caja.getCenter(new THREE.Vector3()));
      gltf.scene.scale.setScalar(2 / tam);
      pivote.add(gltf.scene);
    },
    reiniciar() {
      pivote.rotation.set(0, 0, 0);
      pivote.position.set(0, 0, 0);
      pivote.scale.setScalar(1);
    },
    aplicarGesto(gesto, manos) {
      // TODO (rol 4): rotar con palma_abierta, mover con pellizco, escalar con dos manos,
      // con suavizado. puño -> reiniciar.
      if (gesto === 'puño') this.reiniciar();
    },
  };
}
