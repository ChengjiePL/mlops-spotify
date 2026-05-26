"""Assign alias to latest registered model version."""
import os
import sys

import mlflow


def main():
    model_name = os.environ.get("MODEL_NAME", "workspace.default.spotify_popularity")
    alias = os.environ.get("MODEL_ALIAS", "production")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for {model_name}", file=sys.stderr)
        sys.exit(1)

    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=latest.version,
    )
    print(f"Alias '{alias}' -> {model_name} version {latest.version}")


if __name__ == "__main__":
    main()
