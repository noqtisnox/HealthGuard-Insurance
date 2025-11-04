from fastapi import FastAPI
from .endpoints import inference, training
from fastapi.staticfiles import StaticFiles

from pathlib import Path

BASE_DIR = Path(__file__).parent

app = FastAPI(title="HealthGuard Obesity Risk API")

# Реєстрація Jinja2 Templates (важливо для рендерингу)
# Припускаємо, що app/main.py знаходиться в app/
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Реєстрація ендпоінтів
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
app.include_router(inference.router, prefix="/api/v1/risk", tags=["Inference"])

# NOTE: Ваш root-endpoint тепер може перенаправляти на форму
@app.get("/")
async def redirect_to_form():
    # Повертаємо посилання на HTML-форму
    return {"message": "Go to /api/v1/risk/predict to access the web form."}