"""Register both logged models and set the champion / challenger aliases.

The winner is read from the `winner` tag of the latest compare_sentiment run, so no
run ID is hard-coded and this works on any machine after evaluate.py has been run.
"""
import mlflow
from mlflow import MlflowClient

import config as cfg


def main():
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.EXPERIMENT)
    if experiment is None:
        raise SystemExit("Experiment not found - run evaluate.py first.")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{cfg.COMPARE_RUN_NAME}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No compare run found - run evaluate.py first.")
    compare = runs[0]
    winner = compare.data.tags["winner"]
    run_ids = {"A": compare.data.params["model_a_run_id"], "B": compare.data.params["model_b_run_id"]}
    hf_ids = {"A": cfg.MODEL_A["hf_model_id"], "B": cfg.MODEL_B["hf_model_id"]}
    loser = "B" if winner == "A" else "A"

    versions = {}
    for letter in ("A", "B"):
        mv = mlflow.register_model(f"runs:/{run_ids[letter]}/{cfg.MODEL_PATH}", cfg.REGISTERED_MODEL)
        client.set_model_version_tag(cfg.REGISTERED_MODEL, mv.version, "hf_model_id", hf_ids[letter])
        client.set_model_version_tag(cfg.REGISTERED_MODEL, mv.version, "candidate", letter)
        versions[letter] = mv.version

    client.set_registered_model_alias(cfg.REGISTERED_MODEL, "champion", versions[winner])
    client.set_registered_model_alias(cfg.REGISTERED_MODEL, "challenger", versions[loser])

    print(f"Registered '{cfg.REGISTERED_MODEL}': A=v{versions['A']}, B=v{versions['B']}")
    print(f"champion   -> v{versions[winner]} (Model {winner}, {hf_ids[winner]})")
    print(f"challenger -> v{versions[loser]} (Model {loser}, {hf_ids[loser]})")


if __name__ == "__main__":
    main()
