// Historial de estados del objeto para "Deshacer" (sección 16.3: protección ante un gesto
// mal clasificado). Guarda copias de { pos, rotX, rotY, escala }; no depende de Three.js.

const iguales = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function crearHistorial(maximo = 20) {
  const pila = [];
  return {
    guardar(estado) {
      if (pila.length && iguales(pila.at(-1), estado)) return; // sin cambios desde lo último guardado
      pila.push(structuredClone(estado));
      if (pila.length > maximo) pila.shift();
    },

    // Devuelve el último estado distinto del actual, o null si no hay nada para deshacer.
    // (Un gesto que empezó pero no movió el objeto deja un estado igual al actual.)
    deshacer(actual) {
      while (pila.length) {
        const estado = pila.pop();
        if (!iguales(estado, actual)) return estado;
      }
      return null;
    },

    limpiar() {
      pila.length = 0;
    },

    get cantidad() {
      return pila.length;
    },
  };
}
