"""Descarga las librerías del frontend a frontend/vendor/ para que el visor funcione sin internet.

Plan B del riesgo 2 (sección 17): sin red en el aula, el visor tiene que abrir igual.
Los archivos quedan versionados en el repositorio, así que esto solo hace falta para
actualizar versiones o verificar que no se modificaron.

Uso (desde la raíz):  python scripts/descargar_dependencias.py [--verificar]
"""
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "frontend" / "vendor"

THREE = "0.160.0"
MEDIAPIPE = "0.10.14"
JSDELIVR = "https://cdn.jsdelivr.net/npm"

ARCHIVOS = {
    # three.js (licencia MIT)
    "three/build/three.module.js": f"{JSDELIVR}/three@{THREE}/build/three.module.js",
    "three/examples/jsm/loaders/GLTFLoader.js": f"{JSDELIVR}/three@{THREE}/examples/jsm/loaders/GLTFLoader.js",
    "three/examples/jsm/environments/RoomEnvironment.js":
        f"{JSDELIVR}/three@{THREE}/examples/jsm/environments/RoomEnvironment.js",
    # MediaPipe Tasks Vision (licencia Apache 2.0)
    "mediapipe/vision_bundle.mjs": f"{JSDELIVR}/@mediapipe/tasks-vision@{MEDIAPIPE}/vision_bundle.mjs",
    "mediapipe/wasm/vision_wasm_internal.js": f"{JSDELIVR}/@mediapipe/tasks-vision@{MEDIAPIPE}/wasm/vision_wasm_internal.js",
    "mediapipe/wasm/vision_wasm_internal.wasm":
        f"{JSDELIVR}/@mediapipe/tasks-vision@{MEDIAPIPE}/wasm/vision_wasm_internal.wasm",
    "mediapipe/wasm/vision_wasm_nosimd_internal.js":
        f"{JSDELIVR}/@mediapipe/tasks-vision@{MEDIAPIPE}/wasm/vision_wasm_nosimd_internal.js",
    "mediapipe/wasm/vision_wasm_nosimd_internal.wasm":
        f"{JSDELIVR}/@mediapipe/tasks-vision@{MEDIAPIPE}/wasm/vision_wasm_nosimd_internal.wasm",
    "mediapipe/hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def importaciones_relativas(ruta: Path):
    """Archivos que importa un módulo de three.js con rutas relativas (p. ej. ../utils/...)."""
    texto = ruta.read_text(encoding="utf-8")
    return re.findall(r"""from\s+['"](\.{1,2}/[^'"]+)['"]""", texto)


def descargar(destino: str, url: str):
    ruta = VENDOR / destino
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        ruta.write_bytes(r.read())
    print(f"  {destino:55s} {ruta.stat().st_size / 1e6:6.2f} MB")
    return ruta


def main():
    if "--verificar" in sys.argv:
        registro = (VENDOR / "VERSIONES.md").read_text(encoding="utf-8")
        esperados = dict(re.findall(r"\| `([^`]+)` \| ([0-9a-f]{64}) \|", registro))
        fallas = [d for d, h in esperados.items() if not (VENDOR / d).exists() or sha256(VENDOR / d) != h]
        print("Todo verificado" if not fallas else f"Archivos faltantes o modificados: {fallas}")
        sys.exit(1 if fallas else 0)

    pendientes = dict(ARCHIVOS)
    hechos = {}
    while pendientes:
        destino, url = pendientes.popitem()
        ruta = descargar(destino, url)
        hechos[destino] = url
        # Los módulos de three.js importan otros archivos del mismo paquete: se bajan también
        if destino.startswith("three/examples/") and destino.endswith(".js"):
            for rel in importaciones_relativas(ruta):
                dep_destino = (Path(destino).parent / rel).as_posix()
                dep_destino = str(Path(dep_destino)).replace("\\", "/")
                dep_destino = re.sub(r"[^/]+/\.\./", "", dep_destino)
                if dep_destino not in hechos and dep_destino not in pendientes:
                    pendientes[dep_destino] = f"{JSDELIVR}/three@{THREE}/{dep_destino.removeprefix('three/')}"

    filas = "\n".join(f"| `{d}` | {sha256(VENDOR / d)} |" for d in sorted(hechos))
    (VENDOR / "VERSIONES.md").write_text(f"""# Librerías del frontend (sin internet)

Generado por `scripts/descargar_dependencias.py`. No editar a mano.

| Librería | Versión | Licencia |
|---|---|---|
| three.js | {THREE} | MIT · https://github.com/mrdoob/three.js/blob/dev/LICENSE |
| MediaPipe Tasks Vision | {MEDIAPIPE} | Apache 2.0 · https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE |
| Modelo Hand Landmarker (float16, v1) | 1 | Apache 2.0 · https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |

| Archivo | SHA-256 |
|---|---|
{filas}
""", encoding="utf-8")
    total = sum((VENDOR / d).stat().st_size for d in hechos) / 1e6
    print(f"{len(hechos)} archivos, {total:.1f} MB en {VENDOR}")


if __name__ == "__main__":
    main()
