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
  let fallo = false;

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
      fallo = false;

      rec.onresult = (e) => {
        const resultados = Array.from(e.results, (r) => r[0]?.transcript?.trim() ?? '');
        const texto = resultados.filter(Boolean).join(' ');
        const ultimo = e.results[e.results.length - 1];
        if (ultimo?.isFinal && texto && !finalizado) {
          finalizado = true;
          alFinal(texto);
        } else if (texto) {
          alParcial(texto);
        }
      };
      rec.onerror = (e) => {
        // "no-speech" y "aborted" no son fallas: el usuario no habló o canceló
        if (e.error === 'no-speech' || e.error === 'aborted') {
          fallo = true;
          voz.escuchando = false;
          return alEstado('listo', e.error === 'no-speech' ? 'No escuché nada.' : 'Escucha detenida.');
        }
        const motivo = e.error === 'not-allowed' || e.error === 'service-not-allowed'
          ? 'No hay permiso para usar el micrófono.'
          : e.error === 'network' ? 'El reconocimiento de voz necesita internet.'
          : 'No pude reconocer la voz.';
        fallo = true;
        voz.escuchando = false;
        alEstado('error', motivo);
      };
      rec.onend = () => {
        voz.escuchando = false;
        if (!finalizado && !fallo) alEstado('listo');
      };
      voz.escuchando = true;
      alEstado('escuchando');
      try {
        rec.start();
      } catch (e) {
        fallo = true;
        voz.escuchando = false;
        alEstado('error', e.name === 'NotAllowedError'
          ? 'No hay permiso para usar el micrófono.'
          : 'No se pudo iniciar el reconocimiento de voz.');
      }
    },
    detener() {
      rec?.stop();
    },
  };
  return voz;
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
