import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data import load_data
from features import transform_features
from config import FEATURES, TARGET, DATA_PATH


MODEL_NAME = os.environ.get("MODEL_NAME", "workspace.default.spotify_popularity")


def train():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    experiment_path = os.environ.get(
        "MLFLOW_EXPERIMENT_PATH",
        "/Users/chengjiepenglin06@gmail.com/spotify-popularity",
    )
    mlflow.set_experiment(experiment_path)

    df = load_data(DATA_PATH)
    df = transform_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=420,
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=200, random_state=420)
        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, predictions)
        auc = roc_auc_score(y_valid, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_valid.iloc[:1],
        )

        print(f"Model logged + registered as '{MODEL_NAME}' (run_id={run.info.run_id})")

        metrics_path = Path(__file__).resolve().parent.parent / "metrics.json"
        metrics_path.write_text(json.dumps({
            "accuracy": accuracy,
            "auc": auc,
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "experiment_path": experiment_path,
        }, indent=2))
        print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    train()
# trigger
