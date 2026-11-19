"""MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (Lab 9 style).

This file is shipped with each logged model (code_paths), so predict_from_mlflow.py
can load a model without importing transformers itself.
"""
import mlflow.pyfunc
import pandas as pd

MAX_LENGTH = 128


def build_pipeline(model_id: str):
    """Create the Hugging Face sentiment pipeline (CPU, truncated inputs)."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=model_id,
        device=-1,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def normalise_label(raw_label: str) -> str:
    """Map model-specific labels (POSITIVE / LABEL_1 ...) to positive / negative."""
    label = raw_label.upper()
    if label in ("POSITIVE", "LABEL_1"):
        return "positive"
    if label in ("NEGATIVE", "LABEL_0"):
        return "negative"
    raise ValueError(f"Unexpected label: {raw_label}")


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Takes a DataFrame with a 'text' column, returns text + prediction columns."""

    def __init__(self, hf_model_id: str):
        self.hf_model_id = hf_model_id
        self.pipe = None

    def load_context(self, context):
        self.pipe = build_pipeline(self.hf_model_id)

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        results = [self.pipe(text)[0] for text in texts]
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [normalise_label(r["label"]) for r in results],
                "score": [round(float(r["score"]), 4) for r in results],
                "hf_model_id": self.hf_model_id,
            }
        )
