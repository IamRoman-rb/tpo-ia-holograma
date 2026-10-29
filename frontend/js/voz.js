// Pedidos por voz (sección 3.3: "pedir un objeto escribiendo o hablando").
//
// Usa el reconocimiento de voz del navegador (Web Speech API), en español rioplatense. El texto
// reconocido sigue el mismo camino que un pedido escrito: intención → validación → visor.
//
// Privacidad (sección 16.1): en Chrome y Edge el audio se envía al servicio de reconocimiento del
// navegador (Google o Microsoft), salvo que el navegador ofrezca el reconocimiento en el equipo
// ("processLocally"); en ese caso se usa y la pantalla lo dice. Firefox no tiene la API: el botón se oculta.

export const IDIOMA = 'es-AR';

/** Constructor de reconocimiento del navegador, o null si no hay. */
export function reconocimientoDisponible(global = globalThis) {
  return global.SpeechRecognition ?? global.webkitSpeechRecognition ?? null;
}

/** ¿El navegador puede reconocer en el equipo, sin mandar el audio? (API nueva; si no existe, false). */
export async function puedeReconocerLocal(Reconocimiento) {
  try {
    if (typeof Reconocimiento?.available !== 'function') return false;
    const estado = await Reconocimiento.available({ langs: [IDIOMA], processLocally: true });
    return estado === 'available';
  } catch {
    return false;
  }
}

/**
 * Crea un reconocedor de un solo pedido por vez.
 *   alParcial(texto): lo que se va entendiendo (para mostrarlo mientras se habla)
 *   alFinal(texto):   el pedido completo
 *   alEstado(estado, detalle): 'escuchando' | 'listo' | 'error'
 * Devuelve { escuchar(), detener(), escuchando, local } o null si el navegador no tiene la API.
 */
export function crearVoz({ alParcial = () => {}, alFinal = () => {}, alEstado = () => {}, local = false },
  Reconocimiento = reconocimientoDisponible()) {
  if (!Reconocimiento) return null;
  let rec = null;
  let finalizado = false;

  const voz = {
    escuchando: false,
    local,
    escuchar() {
      if (voz.escuchando) return;
      rec = new Reconocimiento();
      rec.lang = IDIOMA;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      rec.continuous = false;
      if (local && 'processLocally' in rec) rec.processLocally = true;
      finalizado = false;

      rec.onresult = (e) => {
        let texto = '';
        let esFinal = false;
        for (let i = 0; i < e.results.length; i++) {
          texto += e.results[i][0].transcript;
          esFinal ||= e.results[i].isFinal;
        }
        texto = texto.trim();
        if (esFinal && texto && !finalizado) {
          finalizado = true;
          alFinal(texto);
        } else if (texto) {
          alParcial(texto);
        }
      };
      rec.onerror = (e) => {
        // "no-speech" y "aborted" no son fallas: el usuario no habló o canceló
        if (e.error === 'no-speech' || e.error === 'aborted') return alEstado('listo', 'No escuché nada.');
        const motivo = e.error === 'not-allowed' || e.error === 'service-not-allowed'
          ? 'No hay permiso para usar el micrófono.'
          : e.error === 'network' ? 'El reconocimiento de voz necesita internet.'
          : 'No pude reconocer la voz.';
        alEstado('error', motivo);
      };
      rec.onend = () => {
        voz.escuchando = false;
        if (!finalizado) alEstado('listo');
      };
      voz.escuchando = true;
      alEstado('escuchando');
      rec.start();
    },
    detener() {
      rec?.stop();
    },
  };
  return voz;
}

// ---------------------------------------------------------------- voz local (Whisper en el backend)
//
// Con el backend levantado, el audio se graba acá (MediaRecorder) y lo reconoce Whisper en el mismo equipo
// (backend/voz.py): no pasa por Google ni Microsoft. Para saber cuándo empieza y termina de hablar el usuario,
// un detector de voz por energía con umbral adaptativo: mide el volumen (RMS) cada ~50 ms, estima el ruido de
// fondo con su media y su desvío (promedios exponenciales, solo mientras no se habla) y considera voz lo que
// supera la media del ruido en más de K desvíos. Así funciona igual en un aula silenciosa que con un ventilador.

export const DETECTOR = {
  k: 3.5,              // desvíos sobre el ruido para considerar que hay voz
  minimoRms: 0.006,    // por debajo de esto nunca es voz (micrófono casi mudo)
  inicioMs: 120,       // tiene que sonar al menos esto para empezar una frase (descarta golpes)
  silencioMs: 900,     // silencio que cierra la frase
  maximoMs: 12000,     // una frase nunca dura más que esto
  adaptacion: 0.05,    // velocidad con que se actualiza la estadística del ruido
};

/**
 * Detector de voz. actualizar(rms, tMs) devuelve el evento del cuadro: 'silencio', 'inicio' (empezó una frase),
 * 'voz' (sigue), 'fin' (terminó: hay que mandar lo grabado) o 'descartar' (sonó menos que inicioMs).
 */
export function crearDetectorVoz(parametros = DETECTOR) {
  let media = null;
  let varianza = 0;
  let estado = 'silencio';
  let desde = 0;          // cuándo empezó a sonar
  let ultimoSonido = 0;
  const detector = {
    get umbral() {
      return Math.max(parametros.minimoRms, (media ?? 0) + parametros.k * Math.sqrt(varianza));
    },
    get estado() { return estado; },
    actualizar(rms, t) {
      if (media === null) { media = rms; varianza = (rms * 0.5) ** 2; }
      const suena = rms > detector.umbral;
      if (!suena && estado === 'silencio') {
        // solo el ruido de fondo actualiza la estadística: la voz no la contamina
        const d = rms - media;
        media += parametros.adaptacion * d;
        varianza = (1 - parametros.adaptacion) * (varianza + parametros.adaptacion * d * d);
      }
      if (estado === 'silencio') {
        if (suena) { estado = 'posible'; desde = t; ultimoSonido = t; }
        return 'silencio';
      }
      if (suena) ultimoSonido = t;
      if (estado === 'posible') {
        if (!suena) {
          if (t - ultimoSonido > 150) { estado = 'silencio'; return 'descartar'; }
          return 'silencio';
        }
        if (t - desde >= parametros.inicioMs) { estado = 'voz'; return 'inicio'; }
        return 'silencio';
      }
      if (t - ultimoSonido >= parametros.silencioMs || t - desde >= parametros.maximoMs) {
        estado = 'silencio';
        return 'fin';
      }
      return 'voz';
    },
  };
  return detector;
}

/**
 * Micrófono + detector + grabadora. continuo = false: graba una frase y termina (botón del micrófono);
 * continuo = true: escucha sin parar y entrega cada frase (modo manos libres).
 *   alFrase(blob): el audio de una frase completa · alEstado(estado): 'escuchando' | 'hablando' | 'listo' | 'error'
 *   alNivel(rms, umbral): para dibujar el vúmetro
 * Devuelve { detener() }.
 */
export async function escucharMicrofono({ continuo = false, alFrase, alEstado = () => {}, alNivel = () => {} },
  global = globalThis) {
  const flujo = await global.navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  const Contexto = global.AudioContext ?? global.webkitAudioContext;
  const contexto = new Contexto();
  const analizador = contexto.createAnalyser();
  analizador.fftSize = 1024;
  contexto.createMediaStreamSource(flujo).connect(analizador);
  const muestras = new Float32Array(analizador.fftSize);
  const detector = crearDetectorVoz();
  const tipo = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm']
    .find((t) => global.MediaRecorder.isTypeSupported?.(t)) ?? '';
  let grabadora = null;
  let partes = [];
  let activo = true;
  let ocupado = false;   // mientras se cierra una grabación no se mide (el intervalo no se superpone)
  let reloj = null;

  // La grabadora arranca antes de que se detecte la voz (siempre está grabando un tramo): así no se pierde el
  // principio de la primera palabra. Al detectar el inicio se conserva; si era ruido, se descarta.
  const arrancarGrabadora = () => {
    partes = [];
    grabadora = new global.MediaRecorder(flujo, tipo ? { mimeType: tipo } : undefined);
    grabadora.ondataavailable = (e) => { if (e.data.size) partes.push(e.data); };
    grabadora.start(100);
  };
  const cerrarFrase = (entregar) => new Promise((resolver) => {
    const g = grabadora;
    g.onstop = () => {
      if (entregar && partes.length) alFrase(new Blob(partes, { type: g.mimeType || 'audio/webm' }));
      resolver();
    };
    g.stop();
  });

  const detener = async () => {
    if (!activo) return;
    activo = false;
    clearInterval(reloj);
    if (grabadora?.state === 'recording') await cerrarFrase(detector.estado === 'voz');
    flujo.getTracks().forEach((t) => t.stop());
    contexto.close();
    alEstado('listo');
  };

  arrancarGrabadora();
  alEstado('escuchando');
  reloj = setInterval(async () => {
    if (ocupado || !activo) return;
    analizador.getFloatTimeDomainData(muestras);
    let suma = 0;
    for (const v of muestras) suma += v * v;
    const rms = Math.sqrt(suma / muestras.length);
    alNivel(rms, detector.umbral);
    const evento = detector.actualizar(rms, global.performance.now());
    if (evento === 'inicio') alEstado('hablando');
    ocupado = true;
    if (evento === 'descartar' || (evento === 'silencio' && partes.length > 30 && detector.estado === 'silencio')) {
      // ruido o un tramo largo sin voz: se reinicia la grabación para no mandar silencio
      await cerrarFrase(false);
      if (activo) arrancarGrabadora();
    }
    if (evento === 'fin') {
      await cerrarFrase(true);
      if (continuo && activo) { arrancarGrabadora(); alEstado('escuchando'); } else { ocupado = false; return detener(); }
    }
    ocupado = false;
  }, 50);
  return { detener };
}

/** Lee en voz alta una respuesta corta (solo si el pedido vino por voz). */
export function decir(texto, global = globalThis) {
  const sintesis = global.speechSynthesis;
  if (!sintesis || !global.SpeechSynthesisUtterance) return;
  sintesis.cancel();
  const u = new global.SpeechSynthesisUtterance(texto);
  u.lang = IDIOMA;
  sintesis.speak(u);
}
