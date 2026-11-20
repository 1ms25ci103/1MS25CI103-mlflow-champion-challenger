# Champion–Challenger Model Comparison with Hugging Face and MLflow

Skill Lab (MLOps with MLflow) mini project · USN: 1MS25CI103

## 1. Task and models

- **Task:** binary sentiment classification (positive / negative)
- **Dataset:** GLUE SST-2 validation split (`nyu-mll/glue`, config `sst2`), a fixed random sample of **100** sentences (seed 42), saved in `data/eval_inputs.csv`
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters, fast baseline)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M parameters, slower, usually more accurate)
- **New data for step 5.6:** 8 sentences I wrote myself in `data/new_data.csv` (none are from SST-2)

## 2. How to run

```bash
pip install -r requirements.txt

# terminal 1: start the tracking server (keep it running)
python -m mlflow server --host 127.0.0.1 --port 5000

# terminal 2:
python evaluate.py              # runs A and B, logs to MLflow, applies the rule, fills sections 4 and 5
python register_champion.py     # registers both models, sets champion / challenger aliases
python predict_from_mlflow.py   # loads models:/sentiment-model@champion, predicts on data/new_data.csv
```

The first run downloads the models from Hugging Face. `python predict_from_mlflow.py --alias challenger`
loads the other model. To flip the aliases (stretch goal), change them in the MLflow UI (Models → sentiment-model)
and rerun `predict_from_mlflow.py` without editing it. Open http://127.0.0.1:5000 to browse the runs.

Project files: `config.py` (settings and rule constants), `hf_model.py` (MLflow pyfunc wrapper),
`evaluate.py`, `register_champion.py`, `predict_from_mlflow.py`.

## 3. Champion rule (written before the final run)

| Part | Rule |
|---|---|
| Primary metric | **macro-F1** (higher is better) |
| Minimum margin | Challenger must beat the Champion by **at least 0.02 (2 percentage points)**. With 100 samples one sentence is 1 point, so 2 points (2 sentences) is the smallest gap that is not a single-example fluke. |
| Guardrail | Reject the Challenger if its average latency is more than **5x** the Champion's. |
| Tie-breaker | If the difference is inside ±2 points, the **faster model** is the Champion. If the Challenger is worse by more than 2 points, the Champion stays. |
| Direction | accuracy, precision, recall, F1, macro-F1: higher is better. avg_latency_ms, total_inference_s: lower is better. |

These values live in `config.py` and were not changed after seeing results.

## 4. Results

<!-- RESULTS_START -->
| Metric | A (Champion) | B (Challenger) | B - A | Better |
|---|---|---|---|---|
| accuracy | 0.970 | 0.940 | -0.030 | higher |
| precision | 0.966 | 0.948 | -0.018 | higher |
| recall | 0.983 | 0.948 | -0.034 | higher |
| f1 | 0.974 | 0.948 | -0.026 | higher |
| macro_f1 | 0.969 | 0.938 | -0.031 | higher |
| avg_latency_ms | 70.4 | 582.8 | +512.4 | lower |
| total_inference_s | 7.0 | 58.3 | +51.2 | lower |
<!-- RESULTS_END -->

Plots are in `outputs/` and in the MLflow runs. Five examples where the models disagree are in `outputs/disagreements.csv`.

## 5. Decision

<!-- DECISION_START -->
Model A (distilbert-base-uncased-finetuned-sst-2-english) started as the Champion and Model B (siebert/sentiment-roberta-large-english) as the Challenger. On 100 SST-2 validation sentences, A scored a macro-F1 of 0.969 and B scored 0.938, so the Challenger did not improve on the Champion. It was 3.1 percentage points worse, while my rule required it to be at least 2 points better. B was also much slower: 583 ms per sentence against 70 ms for A, about 8 times slower, which breaks my 5x latency guardrail. Either condition alone keeps A as the Champion, so the decision is clear. With only 100 sentences, one sentence is worth 1 point, so a 3-point gap is only about three sentences. That is a small sample, and the gap could shrink or grow on a larger test set. The trade-offs also favour A: it is about five times smaller (roughly 67M vs 355M parameters), needs less memory and runs far faster, while B's extra size gave no accuracy gain here. Latency was measured on a CPU in Colab, so exact times will differ on other machines. Final decision: Model A keeps the champion alias.


<!-- DECISION_END -->

## 6. Limitations and declaration

- Only 100 sentences, one dataset, one seed. A gap of about 2 points is inside the noise at this size.
- SST-2 sentences are short movie-review snippets; results may not carry over to other domains.
- Both models were trained on SST-2-style data (the siebert model on several sentiment datasets, including SST-2), so scores may be optimistic.
- Latency was measured on CPU, one sentence at a time, and depends on the machine.
- The MLflow model stores the Hugging Face model ID and loads the weights from the local Hugging Face cache, so the machine needs internet access the first time.
- **AI assistance:** I used an AI assistant (Claude) to help write the code and the first draft of this README. I have gone through the code and results and can explain them in the viva.
