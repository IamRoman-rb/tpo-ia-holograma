// Orquestador: conecta cámara, gestos, visor y backend.
import { iniciarManos } from './hands.js';
import { clasificarGesto } from './gestures.js';
import { crearVisor } from './viewer.js';
import { pedirObjeto } from './api.js';
import { mostrarGesto, mostrarEstado } from './ui.js';

const UMBRAL_CONFIANZA = 0.8;

const visor = crearVisor(document.getElementById('viewer'));

document.getElementById('reiniciar').addEventListener('click', () => visor.reiniciar());

document.getElementById('pedido').addEventListener('submit', async (e) => {
  e.preventDefault();
  const frase = document.getElementById('frase').value.trim();
  if (!frase) return;
  mostrarEstado('buscando…');
  try {
    const r = await pedirObjeto(frase);
    await visor.cargarModelo(r.modelo_url);
    mostrarEstado(`listo (${r.origen})`);
  } catch (err) {
    console.error(err);
    mostrarEstado('No pude conseguir ese objeto, ¿probás de nuevo?');
  }
});

iniciarManos(document.getElementById('webcam'), (manos) => {
  const { gesto, confianza } = clasificarGesto(manos);
  mostrarGesto(gesto, confianza);
  if (confianza >= UMBRAL_CONFIANZA) visor.aplicarGesto(gesto, manos);
});
