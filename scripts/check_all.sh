#!/usr/bin/env bash
# Prueba rápida pre-defensa. Ejecutar con el backend levantado.
set -u
API="${API:-http://localhost:8000}"
FRONT="${FRONT:-http://localhost:5500}"
fallas=0

check() {
  if eval "$2" >/dev/null 2>&1; then echo "OK    $1"; else echo "FALLA $1"; fallas=$((fallas + 1)); fi
}

check "backend /health"      "curl -fs $API/health"
check "backend /catalog"     "curl -fs $API/catalog"
check "frontend responde"    "curl -fs $FRONT"
check ".env presente"        "test -f \"$(dirname "$0")/../.env\""
check "modelos precargados"  "ls \"$(dirname "$0")/../frontend/models/\"*.glb"

echo
[ "$fallas" -eq 0 ] && echo "Todo listo." || echo "$fallas verificaciones fallaron."
exit "$fallas"
