# TPO IA — Visor 3D controlado con gestos y generado por IA

> **Un visor 3D que se controla con las manos y se crea con una frase, para que los equipos comerciales y técnicos muestren productos o piezas a un cliente sin mouse ni modelado 3D.**

Inteligencia Artificial · UADE · 2do cuatrimestre 2026. La propuesta completa está en [docs/propuesta.md](docs/propuesta.md).

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/` | Propuesta, arquitectura, métricas, uso responsable, decisiones y presentación |
| `frontend/` | Visor Three.js + MediaPipe en el navegador |
| `backend/` | API FastAPI: intención (LLM), búsqueda por similitud, generación 3D, seguridad |
| `ml/` | Captura del dataset de gestos y entrenamiento del MLP |
| `eval/` | Frases de prueba del LLM y resultados de pruebas con usuarios |
| `scripts/` | Verificación pre-defensa y modo offline |

## Levantar el proyecto

Requisitos: Python 3.10+, Chrome o Edge actualizado, cámara web.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env       # completar las claves
uvicorn main:app --reload --port 8000
```

Verificar en <http://localhost:8000/health>.

### Frontend

```bash
cd frontend
python -m http.server 5500
```

Abrir <http://localhost:5500> y permitir el acceso a la cámara.

## Convenciones de trabajo

- Rama `main` protegida; una rama por función (`feature/gestos-mlp`, `feature/llm-intent`).
- *Pull request* con al menos una revisión de otro integrante antes de unir.
- Nunca subir claves de API: van en `.env`, que está en `.gitignore`.
- Las decisiones técnicas se registran en [docs/decisiones.md](docs/decisiones.md).
- Los prompts del producto viven en `backend/prompts/`, no dentro del código.
