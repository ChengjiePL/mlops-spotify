"""Streamlit Cloud entry point. Loads model directly from Databricks UC Model Registry."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import transform_features
from src.config import FEATURES


DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "SpotifyFeatures.csv"
KAGGLE_DATASET = "zaheenhamidani/ultimate-spotify-tracks-db"


def _config(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


DATABRICKS_HOST = _config("DATABRICKS_HOST")
DATABRICKS_TOKEN = _config("DATABRICKS_TOKEN")
MODEL_NAME = _config("MODEL_NAME", "workspace.default.spotify_popularity")
MODEL_ALIAS = _config("MODEL_ALIAS", "production")
KAGGLE_USERNAME = _config("KAGGLE_USERNAME")
KAGGLE_KEY = _config("KAGGLE_KEY")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

if KAGGLE_USERNAME and KAGGLE_KEY:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY


@st.cache_resource(show_spinner="Loading model from Databricks...")
def load_model():
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


def _download_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)


@st.cache_data(show_spinner="Loading dataset...")
def load_songs():
    if not DATA_PATH.exists():
        with st.spinner("Downloading dataset from Kaggle..."):
            _download_dataset()
    df = pd.read_csv(DATA_PATH)
    df_encoded = transform_features(df.copy())
    keep_idx = df.drop_duplicates(subset=["track_id"]).index
    df_raw = df.loc[keep_idx].copy()
    df_encoded = df_encoded.loc[keep_idx]
    df_raw["label_text"] = (
        df_raw["artist_name"].fillna("Unknown").astype(str)
        + " - "
        + df_raw["track_name"].fillna("Unknown").astype(str)
    )
    return df_raw, df_encoded


st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")
st.title("Spotify Hit Predictor")
st.caption(f"Model: `{MODEL_NAME}@{MODEL_ALIAS}`")

model = load_model()
df_raw, df_encoded = load_songs()

genres = sorted(df_raw["genre"].unique())
selected_genre = st.selectbox("Genre filter", ["All"] + genres)

if selected_genre == "All":
    pool = df_raw
else:
    pool = df_raw[df_raw["genre"] == selected_genre]

choice = st.selectbox(
    f"Pick a song ({len(pool):,} available)",
    options=[int(i) for i in pool.index],
    format_func=lambda i: str(pool.loc[i, "label_text"]),
)

if choice is not None:
    track = df_raw.loc[choice]
    features = df_encoded.loc[choice, FEATURES].to_frame().T

    col_info, col_pred = st.columns([2, 1])

    with col_info:
        st.subheader(track["track_name"])
        st.write(f"**Artist:** {track['artist_name']}")
        st.write(f"**Genre:** {track['genre']}")
        st.write(f"**Actual popularity score:** {track['popularity']} / 100")
        actual_label = "popular" if track["popularity"] >= 57 else "not popular"
        st.write(f"**Actual label (>=57 = popular):** {actual_label}")

        st.components.v1.iframe(
            f"https://open.spotify.com/embed/track/{track['track_id']}",
            height=80,
        )

        with st.expander("Audio features"):
            st.dataframe(
                track[[
                    "acousticness", "danceability", "energy", "valence",
                    "instrumentalness", "liveness", "speechiness",
                    "tempo", "duration_ms", "key", "mode", "time_signature",
                ]].to_frame("value")
            )

    with col_pred:
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        label = "popular" if prediction == 1 else "not popular"

        st.subheader("Prediction")
        if prediction == 1:
            st.success(f"{label.upper()}")
        else:
            st.warning(f"{label.upper()}")

        st.metric("Probability of being a hit", f"{probability:.1%}")
        st.progress(probability)

        match = "match" if label == actual_label else "mismatch"
        st.caption(f"Model vs actual: **{match}**")
