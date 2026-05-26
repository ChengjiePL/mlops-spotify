"""Streamlit Cloud entry point. Loads model directly from Databricks UC Model Registry."""
import os

import pandas as pd
import streamlit as st


def _config(key: str, default: str | None = None) -> str | None:
    # Streamlit Cloud → st.secrets, local dev → env var
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


DATABRICKS_HOST = _config("DATABRICKS_HOST")
DATABRICKS_TOKEN = _config("DATABRICKS_TOKEN")
MODEL_NAME = _config("MODEL_NAME", "workspace.default.spotify_popularity")
MODEL_ALIAS = _config("MODEL_ALIAS", "production")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN


@st.cache_resource(show_spinner="Loading model from Databricks…")
def load_model():
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎧")
st.title("Spotify Hit Predictor 🎧")
st.caption(f"Model: `{MODEL_NAME}@{MODEL_ALIAS}`")

model = load_model()

col1, col2 = st.columns(2)

with col1:
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)

with col2:
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)
    duration_ms = st.slider("Duration (ms)", 30_000, 600_000, 200_000)
    key = st.slider("Key", 0, 11, 5)
    mode = st.selectbox("Mode", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor")
    time_signature = st.slider("Time signature", 0, 5, 4)

if st.button("Predict", type="primary"):
    payload = {
        "acousticness": acousticness,
        "danceability": danceability,
        "duration_ms": duration_ms,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "key": key,
        "liveness": liveness,
        "mode": mode,
        "speechiness": speechiness,
        "tempo": tempo,
        "time_signature": time_signature,
        "valence": valence,
    }

    df = pd.DataFrame([payload])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    label = "popular" if prediction == 1 else "not popular"

    if prediction == 1:
        st.success(f"✅ {label.upper()} — prob {probability:.2%}")
    else:
        st.warning(f"❌ {label.upper()} — prob {probability:.2%}")

    st.progress(probability)
    st.json({"prediction": prediction, "probability": probability, "label": label})
