// MediaPipe Hand Landmarker: captura de landmarks en el navegador.
// El video se procesa en el equipo: nunca se envía ni se guarda.
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let detector = null; // se crea una sola vez, aunque se reintente la cámara

async function crearDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opciones = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opciones('GPU'));
  } catch {
    return HandLandmarker.createFromOptions(fileset, opciones('CPU'));
  }
}

// MediaPipe etiqueta la mano suponiendo una imagen espejada (selfie). Como acá la imagen
// llega sin espejar, "Left" corresponde a la mano derecha del usuario.
const lado = (h) => (h?.[0]?.categoryName === 'Left' ? 'derecha' : 'izquierda');

/**
 * Inicia la cámara y llama a onResultados(manos, info) en cada fotograma nuevo.
 * manos: array (0 a 2) de manos, cada una con 21 puntos {x, y, z} en coordenadas de imagen.
 * info: { aspecto, lados, msProcesamiento }
 * Devuelve { ancho, alto } del video.
 */
export async function iniciarManos(video, onResultados) {
  const [det, flujo] = await Promise.all([
    detector ?? crearDetector(),
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } }),
  ]);
  detector = det;
  video.srcObject = flujo;
  await video.play();
  const aspecto = video.videoWidth / video.videoHeight;

  let ultimoTiempo = -1;
  const bucle = () => {
    if (video.srcObject !== flujo) return; // se detuvo o se reemplazó la cámara
    if (video.currentTime !== ultimoTiempo) {
      ultimoTiempo = video.currentTime;
      const t0 = performance.now();
      const res = detector.detectForVideo(video, t0);
      const manos = res.landmarks ?? [];
      onResultados(manos, {
        aspecto,
        lados: (res.handednesses ?? []).map(lado),
        msProcesamiento: performance.now() - t0,
      });
    }
    requestAnimationFrame(bucle);
  };
  bucle();
  return { ancho: video.videoWidth, alto: video.videoHeight };
}
