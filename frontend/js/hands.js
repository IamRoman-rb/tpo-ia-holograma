// MediaPipe Hand Landmarker: captura de landmarks en el navegador.
// El video se procesa en el equipo: nunca se envía ni se guarda.
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// Archivos locales (frontend/vendor/): el visor funciona sin internet (plan B, riesgo 2).
// Se descargan con scripts/descargar_dependencias.py.
const WASM_URL = new URL('../vendor/mediapipe/wasm', import.meta.url).href;
const MODEL_URL = new URL('../vendor/mediapipe/hand_landmarker.task', import.meta.url).href;

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

const CAMARA_QUIETA_MS = 600;    // sin cuadros nuevos este tiempo, el último gesto se suelta (no queda flotando)
const ERRORES_PARA_RECREAR = 30;  // tantos errores seguidos del detector (p. ej., se perdió la GPU): se crea de nuevo

/**
 * Inicia la cámara y llama a onResultados(manos, info) en cada fotograma nuevo.
 * manos: array (0 a 2) de manos, cada una con 21 puntos {x, y, z} en coordenadas de imagen.
 * info: { aspecto, lados, msProcesamiento }
 * alCorte(motivo) avisa si la cámara se desconecta.
 * Devuelve { ancho, alto } del video.
 *
 * Un error en un cuadro (del detector o de quien recibe los resultados) no corta el bucle: se pierde ese cuadro
 * y se avisa que no hay manos, así no queda un gesto congelado en pantalla hasta recargar la página.
 */
export async function iniciarManos(video, onResultados, { alCorte = () => {} } = {}) {
  const [det, flujo] = await Promise.all([
    detector ?? crearDetector(),
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } }),
  ]);
  detector = det;
  const anterior = video.srcObject;
  video.srcObject = flujo;
  anterior?.getTracks?.().forEach((pista) => pista.stop()); // al reintentar, se libera la cámara anterior
  await video.play();
  const aspecto = video.videoWidth / video.videoHeight;

  let ultimoTiempo = -1;
  let ultimoCuadro = performance.now();
  let conManos = false;
  let errores = 0;      // seguidos, del detector
  let errInterfaz = 0;  // en total, al usar los resultados
  const sinManos = { aspecto, lados: [], msProcesamiento: 0 };
  const avisar = (manos, info) => {
    conManos = manos.length > 0;
    onResultados(manos, info);
  };
  const soltar = () => {
    if (!conManos) return;
    try { avisar([], sinManos); } catch (err) { conManos = false; console.error(err); }
  };

  flujo.getVideoTracks()[0]?.addEventListener('ended', () => {
    if (video.srcObject !== flujo) return;
    soltar();
    alCorte('La cámara se desconectó.');
  });

  const bucle = () => {
    if (video.srcObject !== flujo) return; // se detuvo o se reemplazó la cámara
    requestAnimationFrame(bucle);          // primero: pase lo que pase en este cuadro, el siguiente se procesa
    const ahora = performance.now();
    if (!detector) return;                 // se está creando de nuevo
    if (video.currentTime === ultimoTiempo || video.readyState < 2) {
      if (ahora - ultimoCuadro > CAMARA_QUIETA_MS) soltar(); // la cámara dejó de mandar cuadros
      return;
    }
    ultimoTiempo = video.currentTime;
    ultimoCuadro = ahora;
    let res;
    try {
      res = detector.detectForVideo(video, ahora);
      errores = 0;
    } catch (err) {
      errores++;
      if (errores === 1 || errores % 300 === 0) console.error(`Detección de manos (${errores} errores seguidos):`, err);
      soltar();
      if (errores === ERRORES_PARA_RECREAR) {
        detector = null;
        crearDetector().then((d) => { detector = d; }).catch((e) => {
          console.error(e);
          alCorte('Se cortó la detección de manos.');
        });
      }
      return;
    }
    // Un error al usar los resultados (clasificar, dibujar, mover el objeto) solo pierde este cuadro
    try {
      avisar(res.landmarks ?? [], {
        aspecto,
        lados: (res.handednesses ?? []).map(lado),
        msProcesamiento: performance.now() - ahora,
      });
    } catch (err) {
      errInterfaz++;
      if (errInterfaz === 1 || errInterfaz % 300 === 0) console.error(`Gestos (${errInterfaz} errores):`, err);
      soltar();
    }
  };
  bucle();
  return { ancho: video.videoWidth, alto: video.videoHeight };
}
