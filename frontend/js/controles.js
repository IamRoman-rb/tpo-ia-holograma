// Traduce el gesto detectado en acciones del visor (tabla de la sección 12.1, ampliada).
//
// | Gesto            | Acción                                                                  |
// |------------------|-------------------------------------------------------------------------|
// | Señalar          | Cursor en la punta del índice; apuntar 0,8 s a una pieza la elige       |
// | Pellizco         | Mover el objeto; si se pellizca la pieza elegida, se saca solo esa      |
// | Palma abierta    | Rotar                                                                    |
// | Dos pellizcos    | Escalar                                                                  |
// | Victoria (V)     | Con una pieza elegida: aislarla para analizarla. Sin elegir: despiece    |
// | Puño             | Salir del análisis; si no se está analizando, reiniciar la vista         |
// | Pulgar arriba    | Confirmar el pedido que espera confirmación                              |
// | Pulgar abajo     | Deshacer el último cambio                                                |
//
// Cada gesto continuo (pellizco, palma, dos manos) toma un "ancla" en el primer fotograma y luego aplica
// la diferencia respecto de ella. Al cambiar de gesto el ancla se descarta, así el objeto no salta.
//
// La imagen de la cámara se muestra en espejo: cuando la mano va hacia la derecha del usuario, la x de
// MediaPipe disminuye. Por eso se invierte el signo de dx y la x del cursor.
import { crearSeleccionPorPermanencia } from './piezas.js';

const ZONA_MUERTA = 0.002;            // ignora el temblor de los landmarks
const RAD_POR_PANTALLA = 2 * Math.PI; // cruzar toda la pantalla con la palma = una vuelta
const PERMANENCIA_MS = 800;           // apuntar este tiempo a una pieza la elige
const MARGEN_CAMARA = 0.1;            // el 80 % central del cuadro cubre toda la pantalla: se llega a los bordes
const MEMORIA_APUNTADO_MS = 1000;     // al pasar de señalar a pellizcar hay cuadros sin gesto: se recuerda la pieza

const GESTOS_QUE_CAMBIAN = new Set(['pellizco', 'palma_abierta', 'escala_dos_manos', 'puño', 'victoria']);

const puntoPellizco = (mano) => ({ x: (mano[4].x + mano[8].x) / 2, y: (mano[4].y + mano[8].y) / 2 });
const centroPalma = (mano) => mano[9];
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const limitar = (v) => Math.min(1, Math.max(0, v));

/** Punto de la cámara → coordenadas normalizadas de la vista (espejadas). */
export function aVista(p) {
  return {
    u: limitar((1 - p.x - MARGEN_CAMARA) / (1 - 2 * MARGEN_CAMARA)),
    v: limitar((p.y - MARGEN_CAMARA) / (1 - 2 * MARGEN_CAMARA)),
  };
}

function delta(actual, ancla) {
  let dx = -(actual.x - ancla.x);
  let dy = -(actual.y - ancla.y);
  if (Math.abs(dx) < ZONA_MUERTA) dx = 0;
  if (Math.abs(dy) < ZONA_MUERTA) dy = 0;
  return { dx, dy };
}

/**
 * alEvento(tipo, datos) avisa lo que la interfaz tiene que mostrar:
 * 'seleccion' (info de la pieza), 'aislar' (info), 'salir', 'despiece', 'confirmar', 'deshacer'.
 */
export function crearControles(visor, { alEvento = () => {} } = {}) {
  let gestoPrevio = 'ninguno';
  let ancla = null;
  let escalaInicial = 1;
  let piezaTomada = null;      // pieza que se está moviendo sola con el pellizco
  let ultimaApuntada = null;
  let apuntadaEn = -Infinity;
  let progreso = 0;
  const permanencia = crearSeleccionPorPermanencia(PERMANENCIA_MS);

  function alEmpezar(gesto, manos, ahora) {
    // Al empezar un gesto que cambia el objeto se guarda el estado: así "Deshacer"
    // vuelve a como estaba antes de todo el movimiento (sección 16.3)
    if (GESTOS_QUE_CAMBIAN.has(gesto)) visor.guardarEstado();
    switch (gesto) {
      case 'puño':
        if (visor.salirDeAislamiento()) alEvento('salir');
        else visor.reiniciar();
        break;
      case 'pellizco': {
        // Se toma una sola pieza si se pellizca la que está elegida (o la que se acababa de apuntar)
        const { u, v } = aVista(puntoPellizco(manos[0]));
        const elegida = visor.seleccionada();
        const bajo = visor.apuntar(u, v, 1);
        const reciente = ahora - apuntadaEn < MEMORIA_APUNTADO_MS && ultimaApuntada === elegida;
        piezaTomada = elegida && (bajo === elegida || reciente) ? elegida : null;
        if (!piezaTomada) visor.apuntar(null);
        break;
      }
      case 'victoria': {
        const elegida = visor.seleccionada();
        if (visor.aislada()) {
          visor.salirDeAislamiento();
          alEvento('salir');
        } else if (elegida && visor.tienePiezas()) {
          alEvento('aislar', visor.aislar(elegida));
        } else {
          alEvento('despiece');
        }
        break;
      }
      case 'pulgar_arriba':
        alEvento('confirmar');
        break;
      case 'pulgar_abajo':
        alEvento('deshacer', visor.deshacer());
        break;
      case 'escala_dos_manos':
        escalaInicial = visor.escala();
        break;
      default:
        break;
    }
  }

  return function actualizar(gesto, manos, ahora = performance.now()) {
    if (gesto !== gestoPrevio) {
      ancla = null;
      piezaTomada = null;
      alEmpezar(gesto, manos, ahora);
      gestoPrevio = gesto;
    }

    switch (gesto) {
      case 'señalar': {
        const { u, v } = aVista(manos[0][8]);
        const id = visor.apuntar(u, v, progreso);
        if (id) {
          ultimaApuntada = id;
          apuntadaEn = ahora;
        }
        const r = permanencia(id, ahora);
        progreso = r.progreso;
        if (r.elegida) {
          visor.seleccionar(r.elegida);
          alEvento('seleccion', visor.infoPieza(r.elegida));
        }
        return;
      }
      case 'pellizco': {
        const p = puntoPellizco(manos[0]);
        if (ancla) {
          const { dx, dy } = delta(p, ancla);
          if (piezaTomada) visor.moverPieza(piezaTomada, dx, dy);
          else visor.desplazar(dx, dy);
        }
        if (piezaTomada) {
          const { u, v } = aVista(p);
          visor.apuntar(u, v, 1);
        }
        ancla = p;
        break;
      }
      case 'palma_abierta': {
        const p = centroPalma(manos[0]);
        if (ancla) {
          const { dx, dy } = delta(p, ancla);
          visor.rotar(dx * RAD_POR_PANTALLA, dy * RAD_POR_PANTALLA);
        }
        ancla = { x: p.x, y: p.y };
        break;
      }
      case 'escala_dos_manos': {
        const d = dist(puntoPellizco(manos[0]), puntoPellizco(manos[1]));
        if (!ancla) ancla = d;
        else visor.escalarA(escalaInicial * (d / ancla));
        break;
      }
      default:
        // puño, victoria y pulgares actúan una sola vez, al empezar el gesto (arriba)
        break;
    }
    // Fuera de señalar (y del pellizco de una pieza) no hay cursor
    if (!piezaTomada) visor.apuntar(null);
    permanencia(null, ahora);
    progreso = 0;
  };
}
