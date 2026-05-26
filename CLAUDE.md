# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

Spotify song-popularity classifier. RandomForest on Spotify audio features. Training tracked + model registered in **Databricks-hosted MLflow** (no self-host). Served via FastAPI, called from Streamlit UI.

## Setup

Databricks auth (PAT) in `.env` at repo root:

```
DATABRICKS_HOST=https://community.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
MODEL_NAME=spotify-popularity
MODEL_ALIAS=production
MLFLOW_EXPERIMENT_PATH=/Users/<email>/spotify-popularity
```

See `.env.example`. Local Python tooling also reads `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from env (or `~/.databrickscfg` via `databricks configure --token`).

## Commands

### Train (local → push to Databricks)

```bash
source venv/bin/activate
export $(grep -v '^#' .env | xargs)   # or use direnv
cd src && python train.py             # train.py uses sibling imports
```

After train: in Databricks UI → Models → `spotify-popularity` → version N → **set alias `production`** (Databricks moved from Stages to Aliases). API loads `models:/spotify-popularity@production`.

### Run API + UI (Docker)

```bash
docker-compose --env-file .env up -d --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

Note: this codebase uses legacy `docker-compose` (with dash), not `docker compose`.

### Venv requirements

- **Python 3.11** (3.14 lacks wheels for `pyarrow`; mlflow 2.15.1 untested on 3.12+).
- `setuptools<81` pinned in `requirements.txt` — `setuptools>=81` removed `pkg_resources`, which mlflow 2.15.1 still imports.

## Architecture

Three components, glued by Databricks MLflow registry:

1. **Training (`src/`)** — `train.py` does: `data.load_data` → `features.transform_features` → `RandomForestClassifier.fit` → `mlflow.sklearn.log_model(..., registered_model_name="spotify-popularity")`. Tracking URI hardcoded to `"databricks"`; experiment path read from `MLFLOW_EXPERIMENT_PATH` env. Feature list + target name live in `src/config.py` — must stay in sync with `api/main.py`'s `SongFeatures` pydantic schema and `ui/app.py`'s payload (12 features total).

2. **Serving (`api/main.py`)** — FastAPI. `mlflow.set_tracking_uri("databricks")` + `mlflow.pyfunc.load_model("models:/spotify-popularity@production")` at import time. Needs `DATABRICKS_HOST`/`DATABRICKS_TOKEN` in container env. Single `POST /predict` endpoint.

3. **UI (`ui/app.py`)** — Streamlit. 4 sliders; remaining 8 features hardcoded defaults in payload. Posts to `http://api:8000/predict` (compose DNS name; only works inside compose network).

### Feature engineering caveats (`src/features.py`)

- `key` + `time_signature` label-encoded by **order of `.unique()`** — encoding non-deterministic across data refreshes. Inference inputs use raw ints, mapping implicit.
- `popularity` binarized at threshold **57** (≥57 → 1). Target = "popular", not raw popularity.
- `mode`: `Major`→1, `Minor`→0.

### Aliases vs Stages

Databricks deprecated Stages (`/Production`, `/Staging`). Code uses alias syntax: `models:/<name>@<alias>`. If using older workspace that still supports Stages, set `MODEL_ALIAS` env to a version number string and adjust URI accordingly.

### Cold start

`api` loads model from Databricks at startup → first boot pulls artifacts over HTTPS. Container takes longer than self-hosted MLflow path.

## CI/CD

`.github/workflows/train.yml` corre en push a `main` con cambios en `src/**`, `data/**`, `scripts/promote_model.py`, `requirements.txt` o el propio workflow. Pipeline:

1. Check Databricks auth
2. `python src/train.py` → loggea run + registra version en UC
3. `python scripts/promote_model.py` → asigna alias `production` a la versión recién creada

API en Docker carga `models:/<MODEL_NAME>@production` al boot, así que tras workflow exitoso un restart de `api` toma la nueva version.

### GH secrets/vars requeridos

**Secrets** (Settings → Secrets and variables → Actions → Secrets):
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

**Variables** (mismo lugar, pestaña Variables):
- `MLFLOW_EXPERIMENT_PATH` (ej `/Users/<email>/spotify-popularity`)
- `MODEL_NAME` (ej `workspace.default.spotify_popularity`)
- `MODEL_ALIAS` (ej `production`)

## Notes

- `scripts/spotify-song-popularity-prediction.py` and `notebooks/…ipynb` = original Kaggle exploration. Productionized code lives in `src/`, `api/`, `ui/`.
- No tests, linter config, or CI.
- Self-hosted MLflow (Postgres + Docker) was removed in favor of Databricks.
