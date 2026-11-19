"""Run Model A (Champion) and Model B (Challenger) on the SAME eval set, log
everything to MLflow, and apply the Champion rule written in README.md.

Usage:
    python -m mlflow server --host 127.0.0.1 --port 5000      # terminal 1
    python evaluate.py                                        # terminal 2
"""
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import config as cfg
import hf_model

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
EVAL_PATH = DATA_DIR / "eval_inputs.csv"
LABELS = ["negative", "positive"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "macro_f1"]  # higher is better


# ----------------------------------------------------------------- data
def load_eval_set() -> pd.DataFrame:
    """Load data/eval_inputs.csv, or build it once from SST-2 with a fixed seed."""
    if EVAL_PATH.exists():
        return pd.read_csv(EVAL_PATH)
    from datasets import load_dataset

    raw = load_dataset("nyu-mll/glue", "sst2", split="validation").to_pandas()
    sample = raw.sample(n=cfg.EVAL_SAMPLES, random_state=cfg.SEED).reset_index(drop=True)
    df = pd.DataFrame(
        {
            "id": sample.index,
            "text": sample["sentence"].str.strip(),
            "reference": sample["label"].map({0: "negative", 1: "positive"}),
        }
    )
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(EVAL_PATH, index=False)
    return df


# ------------------------------------------------------------ inference
def run_model(model_id: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Predict one example at a time so per-example latency is clean."""
    pipe = hf_model.build_pipeline(model_id)
    pipe("warm up")  # first call is slow; keep it out of the timing
    rows = []
    for text in eval_df["text"]:
        start = time.perf_counter()
        result = pipe(text)[0]
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "prediction": hf_model.normalise_label(result["label"]),
                "score": round(float(result["score"]), 4),
                "latency_ms": round(latency_ms, 2),
            }
        )
    out = eval_df.copy()
    out = pd.concat([out, pd.DataFrame(rows)], axis=1)
    out["correct"] = out["prediction"] == out["reference"]
    return out


def compute_metrics(preds: pd.DataFrame) -> dict:
    y, p = preds["reference"], preds["prediction"]
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, pos_label="positive"),
        "recall": recall_score(y, p, pos_label="positive"),
        "f1": f1_score(y, p, pos_label="positive"),
        "macro_f1": f1_score(y, p, average="macro"),
        "avg_latency_ms": preds["latency_ms"].mean(),
        "total_inference_s": preds["latency_ms"].sum() / 1000,
    }


# ---------------------------------------------------------------- plots
def save_fig(fig, name: str):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig, f"plots/{name}")
    plt.close(fig)


def plot_confusion(preds: pd.DataFrame, title: str, name: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        preds["reference"], preds["prediction"], labels=LABELS, ax=ax, colorbar=False
    )
    ax.set_title(title)
    save_fig(fig, name)


def plot_comparison(ma: dict, mb: dict, preds_a: pd.DataFrame, preds_b: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1]})
    x = range(len(METRIC_NAMES))
    w = 0.38
    ax1.bar([i - w / 2 for i in x], [ma[m] for m in METRIC_NAMES], w, label="A (Champion)")
    ax1.bar([i + w / 2 for i in x], [mb[m] for m in METRIC_NAMES], w, label="B (Challenger)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(METRIC_NAMES)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Quality metrics (higher is better)")
    ax1.legend()
    ax2.bar(["A", "B"], [ma["avg_latency_ms"], mb["avg_latency_ms"]], color=["C0", "C1"])
    ax2.set_title("Avg latency ms (lower is better)")
    save_fig(fig, "champion_vs_challenger_bar.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(preds_a["latency_ms"], bins=25, alpha=0.6, label="A")
    ax.hist(preds_b["latency_ms"], bins=25, alpha=0.6, label="B")
    ax.set_xlabel("per-example latency (ms)")
    ax.set_ylabel("count")
    ax.set_title("Latency distribution")
    ax.legend()
    save_fig(fig, "latency_histogram.png")


# --------------------------------------------------------------- MLflow
def log_pyfunc(model_id: str):
    """Log the wrapper. mlflow 3 uses `name`, mlflow 2 uses `artifact_path`."""
    example_in = pd.DataFrame({"text": ["a great film"]})
    example_out = pd.DataFrame(
        {"text": ["a great film"], "prediction": ["positive"], "score": [0.99], "hf_model_id": [model_id]}
    )
    kwargs = dict(
        python_model=hf_model.HFSentimentModel(model_id),
        code_paths=[str(ROOT / "hf_model.py")],
        signature=infer_signature(example_in, example_out),
        pip_requirements=["mlflow", "transformers", "torch", "pandas"],
    )
    try:
        return mlflow.pyfunc.log_model(name=cfg.MODEL_PATH, **kwargs)
    except TypeError:
        return mlflow.pyfunc.log_model(artifact_path=cfg.MODEL_PATH, **kwargs)


def run_and_log(spec: dict, eval_df: pd.DataFrame):
    short = spec["run_name"].split("_", 2)[2]
    with mlflow.start_run(run_name=spec["run_name"], nested=True):
        mlflow.log_params(
            {
                "hf_model_id": spec["hf_model_id"],
                "task": cfg.TASK,
                "role": spec["role"],
                "max_length": hf_model.MAX_LENGTH,
                "batch_size": 1,
                "eval_samples": len(eval_df),
                "seed": cfg.SEED,
                "device": cfg.DEVICE,
            }
        )
        mlflow.set_tags(
            {
                "student_usn": cfg.USN,
                "library_versions": cfg.library_versions(),
                "dataset_name": cfg.DATASET_NAME,
            }
        )
        preds = run_model(spec["hf_model_id"], eval_df)
        metrics = compute_metrics(preds)
        mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})

        OUT_DIR.mkdir(exist_ok=True)
        pred_file = OUT_DIR / f"predictions_{short}.csv"
        preds.to_csv(pred_file, index=False)
        mlflow.log_artifact(str(pred_file))
        mlflow.log_artifact(str(EVAL_PATH))
        mlflow.log_dict({spec["run_name"]: metrics}, "metrics_summary.json")
        plot_confusion(preds, f"{spec['run_name']}", f"confusion_matrix_{short}.png")
        log_pyfunc(spec["hf_model_id"])
        run_id = mlflow.active_run().info.run_id
    return {"run_id": run_id, "preds": preds, "metrics": metrics, "short": short}


# ------------------------------------------------------------- decision
def decide(ma: dict, mb: dict) -> dict:
    """Apply the Champion rule from README section 3 (no changes after seeing results)."""
    delta = mb[cfg.PRIMARY_METRIC] - ma[cfg.PRIMARY_METRIC]
    latency_ratio = mb["avg_latency_ms"] / ma["avg_latency_ms"]
    faster = "A" if ma["avg_latency_ms"] <= mb["avg_latency_ms"] else "B"
    if latency_ratio > cfg.LATENCY_GUARDRAIL:
        winner, reason = "A", "guardrail"
    elif delta >= cfg.MARGIN:
        winner, reason = "B", "margin"
    elif delta > -cfg.MARGIN:
        winner, reason = faster, "tiebreak"
    else:
        winner, reason = "A", "worse"
    return {"winner": winner, "reason": reason, "delta": delta, "latency_ratio": latency_ratio}


def qualitative_examples(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": preds_a["id"],
            "text": preds_a["text"],
            "reference": preds_a["reference"],
            "pred_A": preds_a["prediction"],
            "pred_B": preds_b["prediction"],
            "score_A": preds_a["score"],
            "score_B": preds_b["score"],
        }
    )
    dis = df[df["pred_A"] != df["pred_B"]].copy()

    def comment(r):
        a_ok, b_ok = r["pred_A"] == r["reference"], r["pred_B"] == r["reference"]
        if a_ok and not b_ok:
            return "Only A correct - replace with your own comment on why B may have failed"
        if b_ok and not a_ok:
            return "Only B correct - replace with your own comment on why A may have failed"
        return "Both wrong - replace with your own comment (sarcasm? mixed sentiment?)"

    dis["comment"] = dis.apply(comment, axis=1)
    return dis.head(5)


def decision_text(spec_a, spec_b, ma, mb, dec, dis_all, n) -> str:
    delta_pp = dec["delta"] * 100
    margin_pp = cfg.MARGIN * 100
    se = math.sqrt(ma["accuracy"] * (1 - ma["accuracy"]) / n) * 100
    a_only = int(((dis_all["pred_A"] == dis_all["reference"])).sum())
    b_only = int(((dis_all["pred_B"] == dis_all["reference"])).sum())
    verdicts = {
        "margin": "The Challenger clears the required margin, so it replaces the Champion.",
        "tiebreak": f"The gain is inside the margin, so the tie-breaker applies and the faster model ({dec['winner']}) is the Champion.",
        "guardrail": "The Challenger is more than 5x slower than the Champion, so the latency guardrail rejects it and the Champion stays.",
        "worse": "The Challenger does not beat the Champion, so the Champion stays.",
    }
    meaning = (
        "a gap this small is within sampling noise and should not be over-read"
        if abs(dec["delta"] * 100) < 2 * se
        else "the gap is larger than the sampling noise, though one subset of this size is still limited evidence"
    )
    return (
        f"Model A ({spec_a['hf_model_id']}) started as Champion and Model B ({spec_b['hf_model_id']}) as Challenger. "
        f"On {n} SST-2 validation sentences, {cfg.PRIMARY_METRIC} was {ma[cfg.PRIMARY_METRIC]:.3f} for A and "
        f"{mb[cfg.PRIMARY_METRIC]:.3f} for B, a difference of {delta_pp:+.1f} percentage points against a required "
        f"margin of {margin_pp:.0f}. {verdicts[dec['reason']]} "
        f"With n={n}, the standard error of accuracy is roughly {se:.1f} points, so {meaning}. "
        f"The models disagreed on {len(dis_all)} sentences: A alone was right on {a_only}, B alone on {b_only}. "
        f"On speed, A averaged {ma['avg_latency_ms']:.0f} ms per example and B {mb['avg_latency_ms']:.0f} ms "
        f"({dec['latency_ratio']:.1f}x A); B is also about five times larger (roughly 355M vs 67M parameters), "
        f"so it costs more memory and compute for whatever accuracy it adds. "
        f"Because the two models make different mistakes, a larger and more varied test set would be needed before trusting a small gap. "
        f"Final decision: Model {dec['winner']} holds the champion alias."
    )


def fill_readme(table_md: str, decision_md: str):
    path = ROOT / "README.md"
    text = path.read_text()
    for tag, body in (("RESULTS", table_md), ("DECISION", decision_md)):
        pattern = rf"(<!-- {tag}_START -->).*?(<!-- {tag}_END -->)"
        text = re.sub(pattern, lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text, flags=re.S)
    path.write_text(text)


# ----------------------------------------------------------------- main
def main():
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)
    eval_df = load_eval_set()
    n = len(eval_df)

    with mlflow.start_run(run_name=cfg.COMPARE_RUN_NAME):
        mlflow.set_tags(
            {
                "student_usn": cfg.USN,
                "library_versions": cfg.library_versions(),
                "dataset_name": cfg.DATASET_NAME,
            }
        )
        mlflow.log_params({"task": cfg.TASK, "eval_samples": n, "seed": cfg.SEED, "device": cfg.DEVICE})

        res_a = run_and_log(cfg.MODEL_A, eval_df)
        res_b = run_and_log(cfg.MODEL_B, eval_df)
        ma, mb = res_a["metrics"], res_b["metrics"]

        dec = decide(ma, mb)
        mlflow.log_params(
            {
                "primary_metric": cfg.PRIMARY_METRIC,
                "margin": cfg.MARGIN,
                "latency_guardrail_x": cfg.LATENCY_GUARDRAIL,
                "model_a_run_id": res_a["run_id"],
                "model_b_run_id": res_b["run_id"],
            }
        )
        mlflow.set_tag("winner", dec["winner"])
        mlflow.set_tag("decision_reason", dec["reason"])
        mlflow.log_metrics(
            {
                "delta_primary_metric": round(dec["delta"], 4),
                "latency_ratio_B_over_A": round(dec["latency_ratio"], 3),
            }
        )

        plot_comparison(ma, mb, res_a["preds"], res_b["preds"])

        # qualitative examples (disagreements)
        all_dis = qualitative_examples(res_a["preds"], res_b["preds"])
        full_dis = res_a["preds"][res_a["preds"]["prediction"] != res_b["preds"]["prediction"]]
        full_dis = pd.DataFrame(
            {
                "reference": full_dis["reference"],
                "pred_A": full_dis["prediction"],
                "pred_B": res_b["preds"].loc[full_dis.index, "prediction"],
            }
        )
        all_dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
        mlflow.log_artifact(str(OUT_DIR / "disagreements.csv"))

        summary = {
            "model_A": {"hf_model_id": cfg.MODEL_A["hf_model_id"], **ma},
            "model_B": {"hf_model_id": cfg.MODEL_B["hf_model_id"], **mb},
            "decision": dec,
        }
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(OUT_DIR / "metrics_summary.json"))

        # README results + decision
        rows = ["| Metric | A (Champion) | B (Challenger) | B - A | Better |", "|---|---|---|---|---|"]
        for m in METRIC_NAMES:
            rows.append(f"| {m} | {ma[m]:.3f} | {mb[m]:.3f} | {mb[m] - ma[m]:+.3f} | higher |")
        rows.append(f"| avg_latency_ms | {ma['avg_latency_ms']:.1f} | {mb['avg_latency_ms']:.1f} | {mb['avg_latency_ms'] - ma['avg_latency_ms']:+.1f} | lower |")
        rows.append(f"| total_inference_s | {ma['total_inference_s']:.1f} | {mb['total_inference_s']:.1f} | {mb['total_inference_s'] - ma['total_inference_s']:+.1f} | lower |")
        text = decision_text(cfg.MODEL_A, cfg.MODEL_B, ma, mb, dec, full_dis, n)
        fill_readme("\n".join(rows), text + f"\n\n_Word count: {len(text.split())}._")
        mlflow.log_artifact(str(ROOT / "README.md"))
        mlflow.log_artifact(str(ROOT / "requirements.txt"))

    print(f"\nWinner: Model {dec['winner']} ({dec['reason']}); delta {cfg.PRIMARY_METRIC} = {dec['delta']:+.4f}")
    print("Next: python register_champion.py")


if __name__ == "__main__":
    main()
