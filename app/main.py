import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import DISCLAIMER, ModelService, stress_category
from app.schema import SleepStressInput, SleepStressOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing")

MODEL_PATH = Path(os.getenv("SLEEP_MODEL_PATH", "models/sleep_model.joblib"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000",
    ).split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.sleep_model = ModelService.load(MODEL_PATH)
        logger.info("Loaded sleep model from %s", MODEL_PATH)
    except FileNotFoundError:
        app.state.sleep_model = None
        logger.warning("Model file %s not found. Train it first.", MODEL_PATH)
    yield


app = FastAPI(
    title="Wellbeing ML API",
    description="Educational wellbeing estimates. Not a diagnosis.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.sleep_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "Wellbeing ML API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.sleep_model is not None}


@app.post("/predict/sleep-stress", response_model=SleepStressOutput)
def predict_sleep_stress(data: SleepStressInput):
    service = app.state.sleep_model
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    try:
        score = service.predict_stress(data)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")
    return SleepStressOutput(
        predicted_stress_level=score,
        category=stress_category(score),
        disclaimer=DISCLAIMER,
    )