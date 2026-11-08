// Dibujo de la detección sobre el recuadro de cámara: esqueleto de cada mano, marco,
// etiqueta con el gesto y ayudas visuales: pellizco, escala con dos manos, flechas al desplazarse con una palma
// y un aro que gira entre las dos palmas al rotar el objeto.
//
// El video se muestra en espejo con CSS. El lienzo no se espeja (si no, el texto saldría
// al revés): en su lugar, cada x se dibuja como (1 - x).
//
// En el modo holograma el lienzo cubre toda la pantalla, encima de la mano real: el esqueleto se dibuja
// fino y luminoso, sin marco, con un aro alrededor de la palma mientras rota y el zoom entre las dos manos.

const CONEXIONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],             // pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],             // índice
  [5, 9], [9, 10], [10, 11], [11, 12],        // medio
  [9, 13], [13, 14], [14, 15], [15, 16],      // anular
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // meñique y palma
];

const COLORES = {
  aplicado: '#2bd4ff',  // el gesto supera el umbral y se aplica al objeto
  detectado: '#ffb84d', // se reconoce un gesto pero todavía no alcanza el umbral
  inactivo: '#c8d6df',
};

const NOMBRES = {
  pellizco: 'Pellizco', palma_abierta: 'Palma abierta', 'puño': 'Puño',
  'señalar': 'Señalar', pulgar_arriba: 'Pulgar arriba', pulgar_abajo: 'Pulgar abajo', victoria: 'Victoria',
  escala_dos_manos: 'Dos manos', ninguno: 'Sin gesto',
};

export function crearOverlay(lienzo) {
  const ctx = lienzo.getContext('2d');
  // El lienzo tiene su propia resolución y se muestra más chico o más grande (con object-fit: cover):
  // k lleva los tamaños de texto y trazo a píxeles de pantalla para que se lean en cualquier tamaño.
  let k = 1;
  let ar = false;

  const punto = (p) => [(1 - p.x) * lienzo.width, p.y * lienzo.height];

  function etiqueta(texto, x, y, color) {
    ctx.font = `600 ${12 * k}px system-ui, sans-serif`;
    const ancho = ctx.measureText(texto).width + 12 * k;
    const alto = 18 * k;
    x = Math.min(Math.max(x, 2), lienzo.width - ancho - 2);
    y = Math.max(y, alto + 2);
    ctx.fillStyle = 'rgba(5, 9, 15, 0.8)';
    ctx.fillRect(x, y - alto, ancho, alto);
    ctx.fillStyle = color;
    ctx.fillRect(x, y - alto, 3 * k, alto);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(texto, x + 7 * k, y - 5 * k);
  }

  function dibujarMano(mano, color, texto) {
    const pts = mano.map(punto);

    // Holograma: trazo blanco fino con un halo del color del estado
    ctx.strokeStyle = ar ? 'rgba(255, 255, 255, 0.85)' : color;
    ctx.lineWidth = (ar ? 1.5 : 2) * k;
    if (ar) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 8 * k;
    }
    ctx.beginPath();
    for (const [a, b] of CONEXIONES) {
      ctx.moveTo(...pts[a]);
      ctx.lineTo(...pts[b]);
    }
    ctx.stroke();

    for (const [i, [x, y]] of pts.entries()) {
      ctx.beginPath();
      ctx.arc(x, y, (i === 4 || i === 8 ? 3.5 : 2.5) * k, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 * k;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // marco de la mano (en el modo holograma, solo la etiqueta)
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const m = 8 * k;
    const [x0, y0] = [Math.min(...xs) - m, Math.min(...ys) - m];
    const [x1, y1] = [Math.max(...xs) + m, Math.max(...ys) + m];
    if (!ar) {
      ctx.setLineDash([4 * k, 3 * k]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 * k;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
    etiqueta(texto, x0, y0, color);
    return pts;
  }

  // Aro que gira alrededor de (cx, cy): el objeto está rotando
  function aro(cx, cy, r, color) {
    const giro = (performance.now() / 600) % (2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * k;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10 * k;
    ctx.beginPath();
    ctx.arc(cx, cy, r, giro, giro + 1.5 * Math.PI);
    ctx.stroke();
    // punta de flecha al final del arco, para que se lea como giro
    const fin = giro + 1.5 * Math.PI;
    const [px, py] = [cx + r * Math.cos(fin), cy + r * Math.sin(fin)];
    const t = 7 * k;
    ctx.beginPath();
    ctx.moveTo(px + t * Math.cos(fin - 0.6), py + t * Math.sin(fin - 0.6));
    ctx.lineTo(px, py);
    ctx.lineTo(px + t * Math.cos(fin - 2.4), py + t * Math.sin(fin - 2.4));
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Flechas en cruz sobre la palma: la mano está desplazando la vista
  function flechas(cx, cy, r, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * k;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8 * k;
    const t = 6 * k;
    for (const ang of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
      const [dx, dy] = [Math.cos(ang), Math.sin(ang)];
      const [x0, y0, x1, y1] = [cx + dx * r * 0.35, cy + dy * r * 0.35, cx + dx * r, cy + dy * r];
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.moveTo(x1 - dx * t - dy * t, y1 - dy * t + dx * t);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 - dx * t + dy * t, y1 - dy * t - dx * t);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
  }

  function lineaPellizco(pts, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * k;
    ctx.beginPath();
    ctx.moveTo(...pts[4]);
    ctx.lineTo(...pts[8]);
    ctx.stroke();
  }

  return {
    /**
     * manos: landmarks de MediaPipe · lados: 'derecha' | 'izquierda' por mano
     * r: resultado del clasificador · aplicado: gesto que se está aplicando al objeto
     * modo: el gesto que los controles aplicaron de verdad ('dos_palmas', 'palma_abierta', …)
     */
    dibujar(manos, lados, r, aplicado, { simulacion = false, etiquetas = null, color = null, holograma = false, zoom = null, modo = null } = {}) {
      ctx.clearRect(0, 0, lienzo.width, lienzo.height);
      ar = holograma;
      k = 1 / Math.max((lienzo.clientWidth || lienzo.width) / lienzo.width, (lienzo.clientHeight || lienzo.height) / lienzo.height);
      if (simulacion) {
        ctx.font = `700 ${11 * k}px system-ui, sans-serif`;
        ctx.fillStyle = 'rgba(255, 184, 77, 0.9)';
        ctx.fillText('SIMULACIÓN · no es una cámara real', 8 * k, lienzo.height - 8 * k);
      }

      const dosManos = r.gesto === 'escala_dos_manos';
      const centros = [];
      const palmas = [];
      manos.slice(0, 2).forEach((mano, i) => {
        const gestoMano = dosManos ? 'pellizco' : (r.porMano?.[i] ?? 'ninguno');
        const colorMano = color ?? (aplicado !== 'ninguno' && (dosManos || i === 0) ? COLORES.aplicado
          : gestoMano !== 'ninguno' ? COLORES.detectado : COLORES.inactivo);
        const lado = lados[i] ? `${lados[i][0].toUpperCase()}${lados[i].slice(1)} · ` : '';
        const texto = etiquetas?.[i] ?? `${lado}${NOMBRES[gestoMano] ?? gestoMano}`;
        const pts = dibujarMano(mano, colorMano, texto);
        if (gestoMano === 'pellizco') lineaPellizco(pts, colorMano);
        const tamano = Math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]);
        if (modo === 'palma_abierta' && i === 0) flechas(...pts[9], 1.3 * tamano, COLORES.aplicado);
        palmas.push({ x: pts[9][0], y: pts[9][1], tamano });
        centros.push([(pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2]);
      });

      // Dos palmas rotando: línea entre las palmas y un aro que gira en el punto medio
      if (modo === 'dos_palmas' && palmas.length === 2) {
        const [a, b] = palmas;
        const [mx, my] = [(a.x + b.x) / 2, (a.y + b.y) / 2];
        ctx.setLineDash([6 * k, 4 * k]);
        ctx.strokeStyle = COLORES.aplicado;
        ctx.lineWidth = 2 * k;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.setLineDash([]);
        aro(mx, my, Math.max(a.tamano, b.tamano) * 0.9, COLORES.aplicado);
        // debajo del aro: arriba están los nombres de cada mano
        etiqueta('Dos palmas · rotar', mx - 52 * k, my + Math.max(a.tamano, b.tamano) * 0.9 + 24 * k, COLORES.aplicado);
      }

      if (dosManos && centros.length === 2) {
        ctx.setLineDash([6 * k, 4 * k]);
        ctx.strokeStyle = aplicado === 'escala_dos_manos' ? COLORES.aplicado : COLORES.detectado;
        ctx.lineWidth = 2 * k;
        ctx.beginPath();
        ctx.moveTo(...centros[0]);
        ctx.lineTo(...centros[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        if (zoom != null) {
          const [mx, my] = [(centros[0][0] + centros[1][0]) / 2, (centros[0][1] + centros[1][1]) / 2];
          etiqueta(`zoom ${zoom.toFixed(2)}×`, mx - 34 * k, my - 8 * k, ctx.strokeStyle);
        }
      }
    },

    limpiar() {
      ctx.clearRect(0, 0, lienzo.width, lienzo.height);
    },
  };
}
