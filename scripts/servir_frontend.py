"""Servidor estático del frontend para desarrollo: como `python -m http.server`, pero sin caché vieja.

`http.server` no manda Cache-Control, así que el navegador puede seguir usando una versión vieja de los
módulos JS aunque se recargue la página. Con "no-cache" pregunta siempre si cambiaron: si no cambiaron,
el servidor contesta 304 y se usa la copia guardada; si cambiaron, baja la nueva.

Uso: python scripts/servir_frontend.py [puerto] [carpeta]
"""
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class SinCacheVieja(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    carpeta = sys.argv[2] if len(sys.argv) > 2 else "frontend"
    manejador = functools.partial(SinCacheVieja, directory=carpeta)
    with ThreadingHTTPServer(("", puerto), manejador) as servidor:
        print(f"Serving HTTP on port {puerto} (sin caché vieja)", flush=True)
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
