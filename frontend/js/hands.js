// MediaPipe Hand Landmarker: captura de landmarks en el navegador.
// El video nunca sale del equipo.
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export async function iniciarManos(video, onResultados) {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 2,
  });

  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
  await video.play();

  let ultimoTiempo = -1;
  const bucle = () => {
    if (video.currentTime !== ultimoTiempo) {
      ultimoTiempo = video.currentTime;
      const res = landmarker.detectForVideo(video, performance.now());
      onResultados(res.landmarks ?? []); // array de manos, cada una con 21 puntos {x, y, z}
    }
    requestAnimationFrame(bucle);
  };
  bucle();
}
