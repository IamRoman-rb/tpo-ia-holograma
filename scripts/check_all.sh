#!/usr/bin/env bash
# Verificación previa a la defensa (secciones 17 y 22 de la propuesta).
#
# Uso (desde la raíz, con el backend y el frontend levantados):
#   bash scripts/check_all.sh            # servicios, plan B sin internet, catálogo y configuración
#   bash scripts/check_all.sh --tests    # además corre todas las pruebas automáticas
#
# FALLA = algo que impide la demo · AVISO = la demo funciona, pero con menos funciones
set -u
cd "$(dirname "$0")/.."

API="${API:-http://localhost:8000}"
FRONT="${FRONT:-http://localhost:5500}"
PY="${PY:-$(command -v python || command -v python3)}"
fallas=0
avisos=0

ok()    { echo "  OK     $1"; }
falla() { echo "  FALLA  $1"; fallas=$((fallas + 1)); }
aviso() { echo "  AVISO  $1"; avisos=$((avisos + 1)); }
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else falla "$1${3:+ → $3}"; fi; }

valor_env() { [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '\r'; }

echo "== Servicios"
if salud=$(curl -fs --max-time 5 "$API/health"); then
  ok "backend responde: $salud"
else
  falla "el backend no responde en $API → cd backend && uvicorn main:app --port 8000"
  salud=""
fi
check "el visor responde" "curl -fs --max-time 5 $FRONT/index.html" "cd frontend && python -m http.server 5500"
check "la herramienta de captura responde" "curl -fs --max-time 5 $FRONT/captura.html"
case "$salud" in
  *'"busqueda":"hibrido"'*) ok "búsqueda con embeddings" ;;
  *'"busqueda":"palabras"'*) aviso "búsqueda solo por palabras: falta sentence-transformers o el modelo en caché" ;;
esac
case "$salud" in
  *'"generacion":"ensamble"'*) ok "generación 3D por piezas con el LLM" ;;
  *'"generacion":"remoto"'*)
    url="$(valor_env GEN3D_URL)"
    check "servidor de generación 3D accesible" "curl -fs --max-time 10 ${url%/}/salud" \
      "volver a correr backend/generador_colab/servidor_generacion.ipynb y actualizar .env"
    ;;
  *) aviso "generación 3D desactivada: los pedidos se resuelven solo con el catálogo" ;;
esac

echo "== Plan B sin internet (riesgo 2)"
check "librerías del frontend completas y sin modificar (frontend/vendor)" \
  "$PY scripts/descargar_dependencias.py --verificar" "python scripts/descargar_dependencias.py"
check "el frontend no depende de ningún CDN" \
  "! grep -rqE 'https://(cdn\.|storage\.googleapis)' frontend/*.html frontend/js"
if ls "$HOME/.cache/huggingface/hub/" 2>/dev/null | grep -q "paraphrase-multilingual-MiniLM-L12-v2"; then
  ok "modelo de embeddings en caché (arrancar el backend con HF_HUB_OFFLINE=1 si no hay red)"
else
  aviso "el modelo de embeddings no está en caché: levantar el backend una vez con internet"
fi

echo "== Catálogo"
faltantes=$("$PY" - <<'EOF'
import json
from pathlib import Path
items = json.loads(Path("frontend/models/catalogo.json").read_text(encoding="utf-8"))
faltan = [it["url"] for it in items if not (Path("frontend") / it["url"]).exists()]
generados = sum(it.get("origen") == "generado" for it in items)
print(f"{len(items)} piezas ({generados} generadas)" if not faltan else "FALTAN " + ", ".join(faltan))
EOF
)
case "$faltantes" in
  FALTAN*) falla "catálogo: $faltantes" ;;
  "") falla "catalogo.json no se pudo leer" ;;
  *) ok "catálogo: $faltantes, todos los .glb presentes" ;;
esac

echo "== Configuración"
if [ -f .env ]; then
  ok ".env presente"
  [ -n "$(valor_env ANTHROPIC_API_KEY)" ] && ok "clave de API del LLM configurada" \
    || aviso "sin ANTHROPIC_API_KEY: los pedidos los interpreta el intérprete por reglas"
else
  aviso "sin .env: sin LLM (se usan reglas) y sin generación 3D"
fi

if [ "${1:-}" = "--tests" ]; then
  echo "== Pruebas automáticas"
  check "frontend (gestos, normalización, búsqueda local, simulación)" "node frontend/tests/gestos.test.mjs"
  check "frontend (piezas, MLP en el navegador, voz, holograma)" "node frontend/tests/piezas.test.mjs && node frontend/tests/mlp.test.mjs && node frontend/tests/voz.test.mjs && node frontend/tests/holograma.test.mjs"
  check "backend (intención, seguridad, búsqueda, generación, API)" \
    "(cd backend && .venv/Scripts/python -m pytest -q tests 2>/dev/null || .venv/bin/python -m pytest -q tests)"
  if [ -d ml/.venv ]; then
    check "ml (normalización igual a la del navegador, formato del dataset)" \
      "(cd ml && .venv/Scripts/python -m pytest -q 2>/dev/null || .venv/bin/python -m pytest -q)"
  else
    aviso "ml/.venv no existe: no se corren las pruebas de ml"
  fi
fi

echo
if [ "$fallas" -eq 0 ]; then
  echo "Listo para la demo ($avisos aviso(s))."
else
  echo "$fallas falla(s) y $avisos aviso(s): revisar antes de la defensa."
fi
exit "$fallas"
