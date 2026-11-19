"""Load the Champion from the MLflow Model Registry by alias and predict on new data.

No Hugging Face import here: prediction goes only through MLflow. The script does not
know which version is the Champion - it asks the registry.

Usage:
    python predict_from_mlflow.py                 # champion
    python predict_from_mlflow.py --alias challenger
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

import config as cfg

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion", choices=["champion", "challenger"])
    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    uri = f"models:/{cfg.REGISTERED_MODEL}@{args.alias}"

    # Which version does the alias point to right now?
    version = MlflowClient().get_model_version_by_alias(cfg.REGISTERED_MODEL, args.alias)
    print(f"Loading {uri}")
    print(f"  registry version : {version.version}")
    print(f"  hf_model_id (tag): {version.tags.get('hf_model_id', 'n/a')}")

    model = mlflow.pyfunc.load_model(uri)
    new_data = pd.read_csv(ROOT / "data" / "new_data.csv")
    result = model.predict(new_data[["text"]])

    result.insert(0, "alias", args.alias)
    result.insert(1, "model_version", version.version)
    print(f"  hf_model_id (from model output): {result['hf_model_id'].iloc[0]}\n")
    print(result[["text", "prediction", "score"]].to_string(index=False, max_colwidth=60))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    result.to_csv(out_dir / "new_data_predictions.csv", index=False)
    print(f"\nSaved {out_dir / 'new_data_predictions.csv'}")


if __name__ == "__main__":
    main()
