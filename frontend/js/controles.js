// Traduce el gesto detectado en acciones del visor (tabla de la sección 12.1).
//
// Cada gesto continuo (pellizco, palma, dos manos) toma un "ancla" en el primer
// fotograma y luego aplica la diferencia respecto de ella. Al cambiar de gesto el
// ancla se descarta, así el objeto no salta.
//
// La imagen de la cámara se muestra en espejo: cuando la mano va hacia la derecha
// del usuario, la x de MediaPipe disminuye. Por eso se invierte el signo de dx.

const ZONA_MUERTA = 0.002;            // ignora el temblor de los landmarks
const RAD_POR_PANTALLA = 2 * Math.PI; // cruzar toda la pantalla con la palma = una vuelta

const GESTOS_QUE_CAMBIAN = new Set(['pellizco', 'palma_abierta', 'escala_dos_manos', 'puño']);

const puntoPellizco = (mano) => ({ x: (mano[4].x + mano[8].x) / 2, y: (mano[4].y + mano[8].y) / 2 });
const centroPalma = (mano) => mano[9];
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function delta(actual, ancla) {
  let dx = -(actual.x - ancla.x);
  let dy = -(actual.y - ancla.y);
  if (Math.abs(dx) < ZONA_MUERTA) dx = 0;
  if (Math.abs(dy) < ZONA_MUERTA) dy = 0;
  return { dx, dy };
}

export function crearControles(visor) {
  let gestoPrevio = 'ninguno';
  let ancla = null;
  let escalaInicial = 1;

  return function actualizar(gesto, manos) {
    if (gesto !== gestoPrevio) {
      ancla = null;
      // Al empezar un gesto que cambia el objeto se guarda el estado: así "Deshacer"
      // vuelve a como estaba antes de todo el movimiento (sección 16.3)
      if (GESTOS_QUE_CAMBIAN.has(gesto)) visor.guardarEstado();
      if (gesto === 'puño') visor.reiniciar();   // se dispara una vez al cerrar la mano
      if (gesto === 'escala_dos_manos') escalaInicial = visor.escala();
      gestoPrevio = gesto;
    }

    switch (gesto) {
      case 'pellizco': {
        const p = puntoPellizco(manos[0]);
        if (ancla) {
          const { dx, dy } = delta(p, ancla);
          visor.desplazar(dx, dy);
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
        // señalar: reservado para seleccionar/marcar (fase 3). ninguno: sin acción.
        break;
    }
  };
}
