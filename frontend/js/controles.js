// Traduce el gesto detectado en acciones del visor (tabla de la sección 12.1, ampliada).
//
// | Gesto            | Acción                                                                  |
// |------------------|-------------------------------------------------------------------------|
// | Señalar          | Cursor en la punta del índice; apuntar 0,8 s a una pieza la elige       |
// | Pellizco         | Mover el objeto; si se pellizca la pieza elegida, se saca solo esa      |
// | Palma abierta    | Desplazarse por el espacio (como arrastrar con el clic izquierdo)        |
// | Dos palmas       | Rotar el objeto siguiendo las manos                                      |
// | Dos pellizcos    | Escalar                                                                  |
// | Victoria (V)     | Con una pieza elegida: aislarla para analizarla. Sin elegir: despiece    |
// | Puño             | Salir del análisis; si no se está analizando, reiniciar la vista         |
// | Pulgar arriba    | Confirmar el pedido que espera confirmación                              |
// | Pulgar abajo     | Deshacer el último cambio                                                |
//
// Cada gesto continuo (pellizco, palma, dos palmas, dos manos) toma un "ancla" en el primer fotograma y luego aplica
// la diferencia respecto de ella. Al cambiar de gesto el ancla se descarta, así el objeto no salta.
//
// La imagen de la cámara se muestra en espejo: cuando la mano va hacia la derecha del usuario, la x de
// MediaPipe disminuye. Por eso se invierte el signo de dx y la x del cursor.
import { crearSeleccionPorPermanencia } from './piezas.js';

const ZONA_MUERTA = 0.002;            // ignora el temblor de los landmarks
// Con las dos palmas, una vuelta entera del objeto = mover las manos un tercio del cuadro de la cámara: se gira
// de un solo movimiento (con las dos manos a la vista el recorrido cómodo es corto)
const RAD_DOS_PALMAS = 3 * 2 * Math.PI;
const PERMANENCIA_MS = 800;           // apuntar este tiempo a una pieza la elige
const MARGEN_CAMARA = 0.1;            // el 80 % central del cuadro cubre toda la pantalla: se llega a los bordes
const MEMORIA_APUNTADO_MS = 1000;     // al pasar de señalar a pellizcar hay cuadros sin gesto: se recuerda la pieza
const CUADROS_DOS_PALMAS = 3;         // cuadros seguidos para entrar o salir de "dos palmas" (la 2.ª mano parpadea)
// Inercia: al soltar las dos palmas girando a más de VEL_INERCIA rad/s (y sin haber frenado antes de soltar), el
// objeto sigue girando a los costados y frena solo. Arriba y abajo no: ese giro está limitado a ±90°.
const VEL_INERCIA = 1.5;
const VEL_INERCIA_MAX = 6 * Math.PI;  // tres vueltas por segundo como mucho
const MS_ULTIMO_MOVIMIENTO = 150;     // si las manos estaban quietas al soltar, no hay inercia

/** Sensibilidad de cada gesto (multiplicadores) e inercia; el panel de ajustes del visor los cambia. */
export const AJUSTES_CONTROLES = { rotar: 1, desplazar: 1, escalar: 1, inercia: true };

// La palma sola mueve la cámara, no el objeto: como el arrastre del mouse, no se guarda para deshacer
const GESTOS_QUE_CAMBIAN = new Set(['pellizco', 'dos_palmas', 'escala_dos_manos', 'puño', 'victoria']);

const puntoPellizco = (mano) => ({ x: (mano[4].x + mano[8].x) / 2, y: (mano[4].y + mano[8].y) / 2 });
const centroPalma = (mano) => mano[9];
const medio = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
// Tamaño de la pantalla en píxeles, para pasarle a la cámara el mismo arrastre que haría el mouse
const pantalla = () => ({ ancho: globalThis.innerWidth || 1280, alto: globalThis.innerHeight || 720 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const limitar = (v) => Math.min(1, Math.max(0, v));

/** Punto de la cámara → coordenadas normalizadas de la vista (espejadas). */
export function aVista(p) {
  return {
    u: limitar((1 - p.x - MARGEN_CAMARA) / (1 - 2 * MARGEN_CAMARA)),
    v: limitar((p.y - MARGEN_CAMARA) / (1 - 2 * MARGEN_CAMARA)),
  };
}

/**
 * Movimiento desde el ancla, ignorando el temblor. En cada eje el ancla solo avanza cuando el movimiento pasa la
 * zona muerta: así un movimiento lento (menos que la zona muerta por cuadro) se acumula y no se pierde.
 * Devuelve { dx, dy, ancla } con la nueva ancla.
 */
export function seguir(actual, ancla) {
  let dx = -(actual.x - ancla.x);
  let dy = -(actual.y - ancla.y);
  if (Math.abs(dx) < ZONA_MUERTA) dx = 0;
  if (Math.abs(dy) < ZONA_MUERTA) dy = 0;
  return { dx, dy, ancla: { x: dx ? actual.x : ancla.x, y: dy ? actual.y : ancla.y } };
}

/**
 * alEvento(tipo, datos) avisa lo que la interfaz tiene que mostrar:
 * 'seleccion' (info de la pieza), 'aislar' (info), 'salir', 'despiece', 'confirmar', 'deshacer'.
 * aPantalla(punto de la cámara) → { u, v } en la vista: por defecto aVista; en el modo holograma, donde se ve el dedo.
 * ajustes() → { rotar, desplazar, escalar, inercia }: se lee en cada cuadro (el panel los cambia en vivo).
 */
export function crearControles(visor, { alEvento = () => {}, aPantalla = aVista, ajustes = () => AJUSTES_CONTROLES } = {}) {
  const ajuste = (clave) => ajustes()?.[clave] ?? AJUSTES_CONTROLES[clave];
  let gestoPrevio = 'ninguno';
  let ancla = null;
  let escalaInicial = 1;
  let piezaTomada = null;      // pieza que se está moviendo sola con el pellizco
  let ultimaApuntada = null;
  let apuntadaEn = -Infinity;
  let progreso = 0;
  let dosPalmas = false;
  let cuadrosCambio = 0;       // cuadros seguidos que piden pasar de una palma a dos (o al revés)
  let velocidad = { x: 0, y: 0 };  // del giro con las dos palmas, en rad/s (promedio de los últimos cuadros)
  let tAnterior = null;
  let ultimoMovimiento = -Infinity;
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
        const { u, v } = aPantalla(puntoPellizco(manos[0]));
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

  // ¿Las dos manos tienen la palma abierta? Con histéresis: un cuadro suelto no cambia de modo
  function hayDosPalmas(gesto, manos, porMano) {
    const ahora = gesto === 'palma_abierta' && (manos?.length ?? 0) >= 2
      && porMano?.length >= 2 && porMano[0] === 'palma_abierta' && porMano[1] === 'palma_abierta';
    if (gesto !== 'palma_abierta') {
      dosPalmas = false;
      cuadrosCambio = 0;
    } else if (ahora === dosPalmas) {
      cuadrosCambio = 0;
    } else if (++cuadrosCambio >= CUADROS_DOS_PALMAS || (ahora && gestoPrevio !== 'palma_abierta')) {
      // Si recién empieza el gesto con las dos palmas a la vista, se rota de entrada
      dosPalmas = ahora;
      cuadrosCambio = 0;
    }
    return dosPalmas; // si se perdió una mano, "dos palmas" sin dos manos no hace nada (ver necesarias)
  }

  /**
   * gesto: el gesto aplicado; manos: los landmarks de cada mano; porMano: el gesto de cada mano (del clasificador),
   * para distinguir una palma abierta (desplazarse) de dos (rotar).
   */
  // Al dejar de rotar con las dos palmas: si venían girando rápido, el objeto sigue girando (inercia)
  function soltarGiro(ahora) {
    const vx = Math.max(-VEL_INERCIA_MAX, Math.min(VEL_INERCIA_MAX, velocidad.x));
    if (ajuste('inercia') && Math.abs(vx) > VEL_INERCIA && ahora - ultimoMovimiento < MS_ULTIMO_MOVIMIENTO) {
      visor.impulsar?.(vx, 0);
    }
    velocidad = { x: 0, y: 0 };
    tAnterior = null;
  }

  /**
   * Devuelve el gesto que se aplicó de verdad: 'dos_palmas' cuando las dos palmas rotan, 'ninguno' si faltaban
   * manos (la interfaz lo muestra como "Palma · desplazar", "Dos palmas · rotar", etc.).
   */
  return function actualizar(gesto, manos, ahora = performance.now(), porMano = null) {
    if (hayDosPalmas(gesto, manos, porMano)) gesto = 'dos_palmas';
    // Un gesto sin las manos que necesita (dos para escalar, una para el resto) no hace nada
    const necesarias = gesto === 'escala_dos_manos' || gesto === 'dos_palmas' ? 2 : gesto === 'ninguno' ? 0 : 1;
    if ((manos?.length ?? 0) < necesarias) gesto = 'ninguno';
    // Pasar de una palma a dos (o al revés) también cambia de gesto: el ancla se descarta y nada salta
    if (gesto !== gestoPrevio) {
      if (gestoPrevio === 'dos_palmas') soltarGiro(ahora);
      ancla = null;
      piezaTomada = null;
      alEmpezar(gesto, manos, ahora);
      gestoPrevio = gesto;
    }

    switch (gesto) {
      case 'señalar': {
        const { u, v } = aPantalla(manos[0][8]);
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
        return gesto;
      }
      case 'pellizco': {
        const p = puntoPellizco(manos[0]);
        if (ancla) {
          const m = seguir(p, ancla);
          if (m.dx || m.dy) {
            if (piezaTomada) visor.moverPieza(piezaTomada, m.dx, m.dy);
            else visor.desplazar(m.dx, m.dy);
          }
          ancla = m.ancla;
        } else ancla = { x: p.x, y: p.y };
        if (piezaTomada) {
          const { u, v } = aPantalla(p);
          visor.apuntar(u, v, 1);
        }
        break;
      }
      case 'palma_abierta': {
        // Desplazarse por el espacio: la escena acompaña a la mano, igual que al arrastrar con el mouse
        const p = centroPalma(manos[0]);
        if (ancla) {
          const m = seguir(p, ancla);
          const { ancho, alto } = pantalla();
          // Mismo factor que el cursor (aVista): la escena se mueve lo mismo que la mano en la pantalla
          const f = 1 / (1 - 2 * MARGEN_CAMARA);
          const k = f * ajuste('desplazar');
          if (m.dx || m.dy) visor.arrastrarCamara(m.dx * ancho * k, -m.dy * alto * k);
          ancla = m.ancla;
        } else ancla = { x: p.x, y: p.y };
        break;
      }
      case 'dos_palmas': {
        // Rotar el objeto siguiendo el punto medio entre las dos palmas
        const p = medio(centroPalma(manos[0]), centroPalma(manos[1]));
        const k = RAD_DOS_PALMAS * ajuste('rotar');
        if (ancla) {
          const m = seguir(p, ancla);
          if (m.dx || m.dy) {
            visor.rotar(m.dx * k, m.dy * k);
            ultimoMovimiento = ahora;
          }
          ancla = m.ancla;
          // Velocidad del giro (para la inercia): promedio exponencial, contando también los cuadros quietos
          const dt = tAnterior == null ? 0 : (ahora - tAnterior) / 1000;
          if (dt > 0) {
            velocidad = { x: 0.6 * velocidad.x + 0.4 * ((m.dx * k) / dt), y: 0.6 * velocidad.y + 0.4 * ((m.dy * k) / dt) };
          }
        } else ancla = p;
        tAnterior = ahora;
        break;
      }
      case 'escala_dos_manos': {
        const d = dist(puntoPellizco(manos[0]), puntoPellizco(manos[1]));
        // Con la sensibilidad, la razón entre las distancias se eleva a una potencia: 2× = el doble de zoom
        if (!ancla) ancla = d;
        else visor.escalarA(escalaInicial * (d / ancla) ** ajuste('escalar'));
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
    return gesto;
  };
}
