# Near-Miss Severity Scoring • Lakehouse + LightGBM (Demo)

This repo is a **portfolio-friendly demo** that mirrors a real project flow:
- **Lakehouse ELT on Delta Lake (Bronze → Silver → Gold)** with **window-based dedup** and **`MERGE INTO`** for idempotent upserts.
- **LightGBM** classification with a small hyperparameter search focusing on **learning rate**, **tree depth / `num_leaves`**, and **class weights** for imbalanced data.
- **Unit tests** for key transforms.
- **Azure ML (v2) command job** YAML to run training in the cloud.
- **Azure DevOps** pipeline stub for CI (lint, tests), with placeholders for AML submit.

> Uses **synthetic data**; pointers to public near-miss/traffic-safety resources are in the bottom of this README.

## Quickstart (local)
```bash
# 1) Create a virtual env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Generate synthetic events (CSV)
python data/synthetic/generate_synthetic_events.py --out data/synthetic/events.csv

# 3) (Optional) Run LightGBM training locally
python src/ml/train_lightgbm.py --train_csv data/synthetic/events.csv --out_dir outputs

# 4) (Optional) Run PySpark dedup + MERGE locally (requires Java + PySpark + delta-spark)
python src/spark/dedup_merge.py   --bronze_csv data/synthetic/events.csv   --silver_path ./lakehouse/silver/events   --gold_path   ./lakehouse/gold/events   --keys store_id,event_id   --ts ts
```

## Lakehouse layout (demo)
- **Bronze**: raw append-only (CSV/Delta)  
- **Silver**: deduplicated, schema-enforced Delta table  
- **Gold**: curated/upserted Delta table for analytics

Dedup logic uses `row_number()` over business keys ordered by a timestamp to keep the latest record; `MERGE INTO` then upserts into **Gold** so replays are **idempotent**.

## LightGBM tuning tidbit (what’s in code)
- `learning_rate`: smaller = slower but often more accurate; paired with higher `n_estimators` + `early_stopping`.
- `num_leaves` / `max_depth`: capacity of trees; too high risks overfit. We tune them jointly.
- `class_weight='balanced'`: compensates for fewer “severe” examples; we still threshold on calibrated probabilities.

## Azure ML (v2) job (optional)
See **azureml/jobs/train.yml** — a minimal **command job** wrapping `train_lightgbm.py`. Edit `compute`, optional `data` inputs, and submit with `az ml job create -f azureml/jobs/train.yml` after configuring your workspace.

## CI (optional)
`azure-devops/azure-pipelines.yml` runs lint/tests. Add service connection + an AML step if you want automatic cloud training on PRs/merges.

## Data & Ethics
This repo uses **synthetic** data only. For real research or interviews, reference public datasets like:
- Near-Miss Incident Database (NIDB) – academic near‑miss work (video).  
- City / DOT crash and traffic datasets (many open portals).

## Repo map
```
src/
  spark/
    dedup_merge.py      # Bronze→Silver dedup + MERGE into Gold (Delta)
  sql/
    dedup_merge.sql     # SQL equivalent for Databricks
  ml/
    train_lightgbm.py   # Tabular LightGBM training + simple tuning
tests/
  test_dedup.py
azureml/
  jobs/train.yml
azure-devops/
  azure-pipelines.yml
env/
  conda.yml
requirements.txt
data/synthetic/
  generate_synthetic_events.py
```
