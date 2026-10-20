// Indicadores en pantalla: gesto detectado, confianza y estado.
const $ = (id) => document.getElementById(id);

export function mostrarGesto(gesto, confianza) {
  $('gesto').textContent = gesto;
  $('confianza').textContent = `${Math.round(confianza * 100)} %`;
}

export function mostrarEstado(texto) {
  $('estado').textContent = texto;
}
