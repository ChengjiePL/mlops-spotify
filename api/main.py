import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = os.environ.get("MODEL_NAME", "workspace.default.spotify_popularity")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI()
model = mlflow.sklearn.load_model(MODEL_URI)


class SongFeatures(BaseModel):
    acousticness: float
    danceability: float
    duration_ms: int
    energy: float
    instrumentalness: float
    key: int
    liveness: float
    mode: int
    speechiness: float
    tempo: float
    time_signature: int
    valence: float


@app.get("/")
def root():
    return {"message": "Spotify MLflow Model API", "model_uri": MODEL_URI}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_song(features: SongFeatures):
    input_data = pd.DataFrame([features.dict()])
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    return {
        "prediction": prediction,
        "probability": probability,
        "label": "popular" if prediction == 1 else "not popular",
    }
