import os
from contextlib import asynccontextmanager
from typing import Any

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_KEY = os.environ.get("MODEL_KEY", "artifacts/current/model.joblib")
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
)
model: Any = None


def download_model() -> None:
    bucket = os.environ.get("ARTIFACT_BUCKET")
    if not bucket:
        raise RuntimeError("ARTIFACT_BUCKET must be set before starting the API")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    boto3.client("s3").download_file(bucket, MODEL_KEY, MODEL_PATH)
    print(f"Downloaded model from s3://{bucket}/{MODEL_KEY}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    download_model()
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)",
        )
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prediction = int(model.predict([req.features])[0])
    return {
        "prediction": prediction,
        "label": "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
