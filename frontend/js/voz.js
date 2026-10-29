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
 * Crea un reconocedor de voz. En modo continuo vuelve a escuchar después de procesar cada pedido.
 *   alParcial(texto): lo que se va entendiendo (para mostrarlo mientras se habla)
 *   alFinal(texto):   el pedido completo
 *   alEstado(estado, detalle): 'escuchando' | 'procesando' | 'listo' | 'error'
 * Devuelve { escuchar(), reanudar(), detener(), activo, escuchando, local } o null si no hay API.
 */
export function crearVoz({
  alParcial = () => {},
  alFinal = () => {},
  alEstado = () => {},
  local = false,
  continuo = false,
  esperaReinicioMs = 350,
}, Reconocimiento = reconocimientoDisponible()) {
  if (!Reconocimiento) return null;
  let rec = null;
  let reinicio = null;
  let finalizado = false;
  let fallo = false;
  let procesando = false;

  const voz = {
    activo: false,
    escuchando: false,
    local,
    _iniciar() {
      if (!voz.activo || voz.escuchando || procesando) return;
      clearTimeout(reinicio);
      reinicio = null;
      rec = new Reconocimiento();
      rec.lang = IDIOMA;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      rec.continuous = false;
      if (local && 'processLocally' in rec) rec.processLocally = true;
      finalizado = false;
      fallo = false;

      rec.onresult = (e) => {
        const resultados = Array.from(e.results, (r) => r[0]?.transcript?.trim() ?? '');
        const texto = resultados.filter(Boolean).join(' ');
        const ultimo = e.results[e.results.length - 1];
        if (ultimo?.isFinal && texto && !finalizado) {
          finalizado = true;
          procesando = true;
          alEstado('procesando');
          rec.stop();
          alFinal(texto);
        } else if (texto) {
          alParcial(texto);
        }
      };
      rec.onerror = (e) => {
        // "no-speech" y "aborted" no son fallas: el usuario no habló o canceló
        if (e.error === 'no-speech' || e.error === 'aborted') {
          voz.escuchando = false;
          if (e.error === 'no-speech') alEstado('listo', 'No escuché nada; sigo escuchando.');
          return;
        }
        const motivo = e.error === 'not-allowed' || e.error === 'service-not-allowed'
          ? 'No hay permiso para usar el micrófono.'
          : e.error === 'network' ? 'El reconocimiento de voz necesita internet.'
          : 'No pude reconocer la voz.';
        fallo = true;
        voz.activo = false;
        voz.escuchando = false;
        alEstado('error', motivo);
      };
      rec.onend = () => {
        voz.escuchando = false;
        if (fallo) return;
        if (procesando) {
          if (!continuo) {
            voz.activo = false;
            procesando = false;
          }
          return;
        }
        if (voz.activo && continuo) {
          if (!finalizado) alEstado('listo', 'Sigo escuchando.');
          reinicio = setTimeout(() => voz._iniciar(), esperaReinicioMs);
        } else {
          voz.activo = false;
          if (!finalizado) alEstado('listo');
        }
      };
      finalizado = false;
      fallo = false;
      voz.escuchando = true;
      alEstado('escuchando');
      try {
        rec.start();
      } catch (e) {
        fallo = true;
        voz.activo = false;
        voz.escuchando = false;
        alEstado('error', e.name === 'NotAllowedError'
          ? 'No hay permiso para usar el micrófono. Pulsa el botón para volver a intentarlo.'
          : 'No se pudo iniciar el reconocimiento de voz.');
      }
    },
    escuchar() {
      voz.activo = true;
      procesando = false;
      voz._iniciar();
    },
    reanudar() {
      procesando = false;
      if (voz.activo && continuo && !voz.escuchando) {
        reinicio = setTimeout(() => voz._iniciar(), esperaReinicioMs);
      }
    },
    detener() {
      voz.activo = false;
      procesando = false;
      clearTimeout(reinicio);
      reinicio = null;
      rec?.stop();
      if (!voz.escuchando) alEstado('listo', 'Micrófono pausado.');
    },
  };
  return voz;
}

/** Lee en voz alta una respuesta corta (solo si el pedido vino por voz). */
export function decir(texto, global = globalThis) {
  const sintesis = global.speechSynthesis;
  if (!sintesis || !global.SpeechSynthesisUtterance) return Promise.resolve();
  sintesis.cancel();
  const u = new global.SpeechSynthesisUtterance(texto);
  u.lang = IDIOMA;
  return new Promise((resolver) => {
    u.onend = resolver;
    u.onerror = resolver;
    sintesis.speak(u);
  });
}
