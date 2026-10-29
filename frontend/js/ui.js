// Indicadores en pantalla: gesto detectado, confianza, estado, origen del modelo.
import { nombreAccion, nombreMotor } from './motores.js';

const $ = (id) => document.getElementById(id);

const NOMBRES_GESTO = {
  pellizco: 'Pellizco',
  palma_abierta: 'Palma abierta',
  'puño': 'Puño',
  'señalar': 'Señalar',
  pulgar_arriba: 'Pulgar arriba',
  pulgar_abajo: 'Pulgar abajo',
  victoria: 'Victoria',
  escala_dos_manos: 'Dos manos',
  ninguno: '—',
};

export function mostrarGesto(gesto, confianza, umbral) {
  $('gesto').textContent = NOMBRES_GESTO[gesto] ?? gesto;
  $('confianza').textContent = confianza ? `${Math.round(confianza * 100)} %` : '—';
  const barra = $('barra-confianza');
  barra.style.width = `${Math.round(confianza * 100)}%`;
  barra.classList.toggle('baja', confianza < umbral);
}

// Los mensajes de error van al aviso sobre el cuadro de pedido; el panel muestra solo el estado corto.
export function mostrarEstado(texto, tipo = 'normal') {
  const el = $('estado');
  const error = tipo === 'error';
  el.textContent = error ? 'atención' : texto;
  el.classList.toggle('error', error);
  $('aviso').textContent = error ? texto : '';
  $('aviso').hidden = !error;
}

// Recuadro de cámara. estado: 'iniciando' | 'activa' | 'simulacion' | 'error'
const TITULOS_CAMARA = {
  iniciando: 'Cámara · iniciando…', activa: 'Cámara en vivo', simulacion: 'Simulación', error: 'Sin cámara',
};

export function mostrarCamara(estado, mensaje = '') {
  $('camara').dataset.estado = estado;
  $('camara-titulo').textContent = TITULOS_CAMARA[estado];
  $('camara-vacia').hidden = estado !== 'error';
  $('camara-detener').hidden = estado !== 'simulacion';
  $('camara-mensaje').textContent = mensaje;
  if (estado !== 'activa' && estado !== 'simulacion') $('camara-fps').textContent = '';
}

export function mostrarFps(fps, msProcesamiento) {
  const ms = msProcesamiento ? ` · ${msProcesamiento.toFixed(0)} ms` : '';
  $('camara-fps').textContent = `${fps.toFixed(0)} fps${ms}`;
}

export function alternarTamanoCamara() {
  const grande = $('camara').classList.toggle('grande');
  $('camara-tamano').setAttribute('aria-label', grande ? 'Achicar el recuadro' : 'Agrandar el recuadro');
  $('camara-tamano').title = grande ? 'Achicar' : 'Agrandar';
}

// "Entendí: mostrar «motor» · vista trasera · ×0.67 — 93 % (LLM)"
export function mostrarInterpretacion(i) {
  const partes = [`<strong>${nombreAccion(i.accion)}</strong>`];
  if (i.objeto) partes.push(`«${escapar(i.objeto)}»`);
  if (i.parametros?.vista) partes.push(`vista ${i.parametros.vista}`);
  if (i.parametros?.factor_escala) partes.push(`×${i.parametros.factor_escala}`);
  if (i.parametros?.motor) partes.push(`${escapar(nombreMotor(i.parametros.motor))} para los ${i.parametros.ambito ?? 'pedidos'}`);
  if (i.parametros?.pregunta) partes.push({ que_motor: 'qué modelo', confianza: 'qué confianza', explicar: 'por qué' }[i.parametros.pregunta]);
  const el = $('interpretacion');
  el.innerHTML = `Entendí: ${partes.join(' · ')} — ${Math.round(i.confianza * 100)} % (${escapar(nombreMotor(i.motor))})`;
  el.hidden = false;
}

const escapar = (t) => t.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

/** Si hay un pedido esperando confirmación, lo confirma (gesto de pulgar arriba). Devuelve si había uno. */
export function confirmarPendiente() {
  if ($('confirmacion').hidden) return false;
  $('confirmar-si').click();
  return true;
}

export function pedirConfirmacion() {
  const caja = $('confirmacion');
  caja.hidden = false;
  return new Promise((resolver) => {
    const responder = (valor) => () => {
      caja.hidden = true;
      $('confirmar-si').onclick = $('confirmar-no').onclick = null;
      resolver(valor);
    };
    $('confirmar-si').onclick = responder(true);
    $('confirmar-no').onclick = responder(false);
  });
}

export function mostrarOrigen(nombre, origen) {
  const el = $('origen');
  el.hidden = false;
  el.classList.toggle('generado', origen === 'generado');
  $('origen-nombre').textContent = nombre;
  $('origen-etiqueta').textContent = origen === 'generado'
    ? 'Generado por IA · no apto para fabricación sin revisión'
    : 'Del catálogo';
}

export function mostrarCatalogo(items, alElegir) {
  const lista = $('lista-catalogo');
  lista.replaceChildren(...items.map((item) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = item.nombre;
    if (item.origen === 'generado') {
      const marca = document.createElement('span');
      marca.className = 'marca-ia';
      marca.textContent = 'IA';
      marca.title = 'Generado por IA';
      b.append(' ', marca);
    }
    b.title = item.descripcion;
    b.dataset.id = item.id;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => alElegir(item));
    li.append(b);
    return li;
  }));
}

export function marcarSeleccionado(id) {
  for (const b of $('lista-catalogo').querySelectorAll('button')) {
    b.setAttribute('aria-pressed', String(b.dataset.id === id));
  }
}

export function alternarAyuda() {
  const ayuda = $('ayuda');
  ayuda.hidden = !ayuda.hidden;
  $('boton-ayuda').setAttribute('aria-expanded', String(!ayuda.hidden));
}

export function alternarDepuracion() {
  $('depuracion').hidden = !$('depuracion').hidden;
}

let lineasGesto = [];

// r e info son null cuando solo se actualizan las latencias (después de un pedido)
export function mostrarDepuracion(r, info, latencias = []) {
  if (r) {
    lineasGesto = [`procesamiento: ${info.msProcesamiento.toFixed(1)} ms`, `crudo: ${r.crudo ?? '—'}`];
    (r.metricas ?? []).forEach((m, i) => {
      lineasGesto.push(`— mano ${i + 1}`);
      for (const [k, v] of Object.entries(m)) lineasGesto.push(`${k}: ${v.toFixed(2)}`);
    });
  }
  const el = $('depuracion');
  if (el.hidden) return;
  const extra = latencias.length ? ['— latencias (mediana)', ...latencias] : [];
  el.textContent = [...lineasGesto, ...extra].join('\n');
}

// Panel de piezas. lista: [{ id, nombre, descripcion, fija, desmontada, seleccionada }]
export function mostrarPiezas(lista, alElegir) {
  $('piezas').hidden = lista.length < 2;
  if (lista.length < 2) return;
  const desmontables = lista.filter((p) => !p.fija);
  $('piezas-cuenta').textContent = `${desmontables.filter((p) => p.desmontada).length}/${desmontables.length} afuera`;
  $('lista-piezas').replaceChildren(...lista.map((p) => {
    const li = document.createElement('li');
    li.textContent = p.fija ? `${p.nombre} (fija)` : p.nombre;
    li.classList.toggle('desmontada', p.desmontada);
    li.classList.toggle('fija', p.fija);
    li.classList.toggle('seleccionada', p.seleccionada);
    li.addEventListener('click', () => alElegir(p.id));
    return li;
  }));
  const elegida = lista.find((p) => p.seleccionada);
  $('pieza-descripcion').textContent = elegida
    ? `${elegida.nombre}: ${elegida.descripcion || 'sin descripción'}`
    : 'Tocá una pieza en la lista o en el modelo para ver qué es.';
}

// Panel de análisis de una pieza aislada. info: { nombre, descripcion, materiales, medidas, orden, fija }
export function mostrarAnalisis(info) {
  if (!info) return ocultarAnalisis();
  $('analisis').hidden = false;
  $('analisis-nombre').textContent = info.nombre;
  $('analisis-descripcion').textContent = info.descripcion || 'Sin descripción.';
  $('analisis-material').textContent = info.materiales?.length ? info.materiales.join(', ') : 'No informado';
  $('analisis-medidas').textContent = info.medidas;
  $('analisis-orden').textContent = info.fija
    ? 'Pieza fija: las demás se desmontan a su alrededor'
    : `Sale en el paso ${info.orden} del desmontaje`;
}

export function ocultarAnalisis() {
  $('analisis').hidden = true;
}

export function fijarExplosion(fraccion) {
  $('explosion').value = Math.round(fraccion * 100);
}
