// Actividad de la IA en tiempo real. main.js publica cada fotograma clasificado (qué vio la red neuronal o las
// reglas) y cada paso de un pedido (qué se le mandó al LLM, qué respondió, cuánto tardó). Lo reciben el panel
// "Actividad IA" del visor y, por un BroadcastChannel, la ventana actividad.html si está abierta (sirve para
// mostrarlo en otra pantalla). El canal es local al navegador: nada de esto sale del equipo.
export const CANAL = 'holograma-actividad';
const VIGENCIA_VENTANA_MS = 4000; // la ventana avisa que sigue abierta cada 1,5 s

export function crearActividad() {
  const oyentes = new Set();
  let canal = null;
  let ventanaVista = -Infinity;
  try {
    canal = new BroadcastChannel(CANAL);
    canal.onmessage = (e) => {
      if (e.data?.tipo !== 'ventana') return;
      ventanaVista = performance.now();
      canal.postMessage({ tipo: 'hola' }); // la ventana sabe que el visor está abierto aunque no haya cámara
    };
  } catch { /* sin BroadcastChannel: solo el panel del visor */ }

  const hayVentana = () => performance.now() - ventanaVista < VIGENCIA_VENTANA_MS;

  return {
    publicar(mensaje) {
      for (const f of oyentes) f(mensaje);
      if (canal && hayVentana()) canal.postMessage(mensaje);
    },
    escuchar(f) {
      oyentes.add(f);
      return () => oyentes.delete(f);
    },
  };
}

/** Para actividad.html: recibe los mensajes del visor y le avisa que la ventana está abierta. */
export function escucharDesdeVentana(f) {
  const canal = new BroadcastChannel(CANAL);
  canal.onmessage = (e) => f(e.data);
  const avisar = () => canal.postMessage({ tipo: 'ventana' });
  avisar();
  setInterval(avisar, 1500);
}
