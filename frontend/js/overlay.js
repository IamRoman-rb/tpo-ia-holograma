// Dibujo de la detección sobre el recuadro de cámara: esqueleto de cada mano, marco,
// etiqueta con el gesto y ayudas visuales (pellizco y escala con dos manos).
//
// El video se muestra en espejo con CSS. El lienzo no se espeja (si no, el texto saldría
// al revés): en su lugar, cada x se dibuja como (1 - x).

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
  'señalar': 'Señalar', escala_dos_manos: 'Dos manos', ninguno: 'Sin gesto',
};

export function crearOverlay(lienzo) {
  const ctx = lienzo.getContext('2d');
  // El lienzo tiene la resolución del video (640 px) pero se muestra más chico: k lleva
  // los tamaños de texto y trazo a píxeles de pantalla para que se lean en cualquier tamaño.
  let k = 1;

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

    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * k;
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

    // marco de la mano
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const m = 8 * k;
    const [x0, y0] = [Math.min(...xs) - m, Math.min(...ys) - m];
    const [x1, y1] = [Math.max(...xs) + m, Math.max(...ys) + m];
    ctx.setLineDash([4 * k, 3 * k]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1 * k;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
    etiqueta(texto, x0, y0, color);
    return pts;
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
     */
    dibujar(manos, lados, r, aplicado, { simulacion = false, etiquetas = null, color = null } = {}) {
      ctx.clearRect(0, 0, lienzo.width, lienzo.height);
      k = lienzo.width / (lienzo.clientWidth || lienzo.width);
      if (simulacion) {
        ctx.font = `700 ${11 * k}px system-ui, sans-serif`;
        ctx.fillStyle = 'rgba(255, 184, 77, 0.9)';
        ctx.fillText('SIMULACIÓN · no es una cámara real', 8 * k, lienzo.height - 8 * k);
      }

      const dosManos = r.gesto === 'escala_dos_manos';
      const centros = [];
      manos.slice(0, 2).forEach((mano, i) => {
        const gestoMano = dosManos ? 'pellizco' : (r.porMano?.[i] ?? 'ninguno');
        const colorMano = color ?? (aplicado !== 'ninguno' && (dosManos || i === 0) ? COLORES.aplicado
          : gestoMano !== 'ninguno' ? COLORES.detectado : COLORES.inactivo);
        const lado = lados[i] ? `${lados[i][0].toUpperCase()}${lados[i].slice(1)} · ` : '';
        const texto = etiquetas?.[i] ?? `${lado}${NOMBRES[gestoMano] ?? gestoMano}`;
        const pts = dibujarMano(mano, colorMano, texto);
        if (gestoMano === 'pellizco') lineaPellizco(pts, colorMano);
        centros.push([(pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2]);
      });

      if (dosManos && centros.length === 2) {
        ctx.setLineDash([6 * k, 4 * k]);
        ctx.strokeStyle = aplicado === 'escala_dos_manos' ? COLORES.aplicado : COLORES.detectado;
        ctx.lineWidth = 2 * k;
        ctx.beginPath();
        ctx.moveTo(...centros[0]);
        ctx.lineTo(...centros[1]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    },

    limpiar() {
      ctx.clearRect(0, 0, lienzo.width, lienzo.height);
    },
  };
}
