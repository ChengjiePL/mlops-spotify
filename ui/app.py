import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.title("Spotify Hit Predictor")

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

if st.button("Predict"):
    payload = {
        "acousticness": acousticness,
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "duration_ms": duration_ms,
        "instrumentalness": instrumentalness,
        "key": key,
        "liveness": liveness,
        "mode": mode,
        "speechiness": speechiness,
        "tempo": tempo,
        "time_signature": time_signature,
    }

    res = requests.post(f"{API_URL}/predict", json=payload)
    data = res.json()

    label = data.get("label", "?")
    prob = data.get("probability", 0.0)

    if data.get("prediction") == 1:
        st.success(f"{label.upper()} - prob {prob:.2%}")
    else:
        st.warning(f"{label.upper()} - prob {prob:.2%}")

    st.progress(prob)
    st.json(data)
