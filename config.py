"""Shared settings. The Champion rule constants below are the SAME ones written in
README.md section 3 - they were fixed before the final run and must not change."""
import os
from importlib.metadata import version

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
REGISTERED_MODEL = "sentiment-model"
TASK = "sentiment"
USN = "1MS25CI103"
DATASET_NAME = "nyu-mll/glue sst2 (validation subset)"
SEED = 42
EVAL_SAMPLES = 100
DEVICE = "cpu"
MODEL_PATH = "model"  # artifact path of the logged model inside each run

# Model A = Champion (fast baseline), Model B = Challenger (larger, usually more accurate)
MODEL_A = {
    "run_name": "A_sentiment_distilbert-sst2",
    "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
    "role": "champion",
}
MODEL_B = {
    "run_name": "B_sentiment_roberta-large",
    "hf_model_id": "siebert/sentiment-roberta-large-english",
    "role": "challenger candidate",
}
COMPARE_RUN_NAME = f"compare_{TASK}"

# ---- Champion rule (see README section 3) ----
PRIMARY_METRIC = "macro_f1"      # higher is better
MARGIN = 0.02                    # Challenger must beat Champion by >= 2 percentage points
LATENCY_GUARDRAIL = 5.0          # reject Challenger if avg latency > 5x Champion's
# Tie-breaker: if the gain is inside the margin, the faster model wins (the Champion stays).


def library_versions() -> str:
    names = ["transformers", "torch", "mlflow"]
    out = []
    for n in names:
        try:
            out.append(f"{n}={version(n)}")
        except Exception:
            out.append(f"{n}=not-installed")
    return ", ".join(out)
