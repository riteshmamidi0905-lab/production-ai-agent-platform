> **▶ [Live demo](https://riteshmamidi0905-lab.github.io/demos/mlops-platform.html)** — the interactive monitoring dashboard this project generates.

<h1 align="center">MLOps Platform</h1>

<p align="center">
  <b>The infrastructure around a model — not just the model.</b><br>
  Feature store · experiment tracking · model registry · serving · drift monitoring
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/tests-12%20passing-brightgreen">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

An **end-to-end machine-learning platform** that takes a model from training to
production and keeps watching it: a **feature store**, **experiment tracking**,
a multi-candidate **training pipeline**, a **model registry** with stage
promotion, a **FastAPI serving** layer, and **data + prediction drift
monitoring** — all wired together behind a CLI. Pure Python/scikit-learn, no
external services required.

![dashboard](docs/dashboard.png)

## The lifecycle it manages

```
train ─► track experiments ─► register best ─► promote to Production ─► serve ─► monitor for drift
```

- **Feature store** — named feature views (with transforms) shared by training
  and serving, so features never skew between the two.
- **Experiment tracking** — every candidate model logs params, metrics, and a
  serialized artifact; the best-by-metric run is selectable (a mini-MLflow).
- **Model registry** — versions + stages (`None → Staging → Production →
  Archived`); promoting to Production auto-archives the previous one, so there's
  always exactly one live model.
- **Serving** — a FastAPI service loads the Production model and scores rows via
  `/predict`; the model bundle carries its own scaler + feature view (no skew).
- **Monitoring** — **PSI** and **KS** per feature detect covariate drift; a
  prediction monitor **alerts** when the model's score distribution shifts. The
  serving layer feeds it live: every `/predict` logs its scores, so
  `/monitoring/predictions` shows real served-traffic drift, not just a batch job.

## Production serving

The serving layer is built like a real model server, not a toy endpoint:

- **Liveness vs readiness** — `/health` says the process is up; `/ready` returns
  `503` until a Production model is actually registered and loadable (the probe
  a load balancer should gate on).
- **API-key auth** — set `MLOPS_API_KEYS`; `/predict` then requires `X-API-Key`.
- **Prometheus metrics** — `/metrics` exposes request rate, latency, rows scored,
  the **served-score distribution** (drift early-warning), and the live model
  version; `docker compose up` wires a Prometheus scraper automatically.
- **Structured logs** — every request logs JSON (method, path, status, latency).
- **Version in every response** — `/predict` returns the `model_version` that
  produced the scores, so predictions are traceable back to a registry entry.

## Results (seeded run)

```
Candidates  : logreg AUC 0.901 · grad_boost 0.887 · random_forest 0.878
Registered  : churn v1 (logreg) → Production
Drift check : 5/5 features drift under covariate shift (PSI > 0.2)  → ALERT
Tests       : 12 passing (serving API, readiness, auth, live monitoring)
```

## Quickstart

```bash
git clone https://github.com/riteshmamidi0905-lab/mlops-platform.git
cd mlops-platform
pip install -e ".[dev]"

mlops --root workspace train         # train candidates, register + promote best
mlops --root workspace monitor       # score a drifted batch, report PSI/KS + alert
mlops --root workspace dashboard --out docs/dashboard.html
pytest -q                            # 12 passing tests

# serve the Production model (health/ready/metrics/predict):
mlops --root workspace serve --port 8000
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"rows":[{"tenure_months":3,"monthly_charges":120,"num_support_tickets":6,"usage_gb":10,"days_since_login":40}]}'
curl -s localhost:8000/metrics                 # Prometheus telemetry
curl -s localhost:8000/monitoring/predictions  # live served-score drift

# or the whole thing (train → serve) + Prometheus in Docker:
docker compose up --build                      # API :8000 · Prometheus :9090
```

## Python API

```python
from mlops import (make_dataset, ExperimentTracker, TrainingPipeline,
                   ModelRegistry, Predictor, detect_drift)

ds = make_dataset()
tracker = ExperimentTracker("mlruns")
runs, _ = TrainingPipeline(tracker).run(ds.X, ds.y)      # trains + logs candidates
best = tracker.best_run("roc_auc")

reg = ModelRegistry("registry_store")
mv = reg.register("churn", best.run_id, best.artifact_path, best.metrics)
reg.transition_stage("churn", mv.version, "Production")   # promote

pred = Predictor(reg, "churn")                            # serves Production model
```

## Project layout

```
mlops/
  data.py            synthetic churn data (+ drifted batches)
  features/          transforms · feature store (views shared train↔serve)
  training/          experiment tracker · training pipeline (ModelBundle)
  registry/          model registry with versions + stage promotion
  serving/           Predictor · FastAPI API · observability · auth
  monitoring/        PSI/KS drift detection · prediction monitor
  cli.py             train · monitor · dashboard · serve
  viz.py             HTML dashboard
tests/               12 passing tests
docs/architecture.md · Dockerfile · docker-compose.yml (+ Prometheus)
deploy/prometheus.yml · .github/workflows/ci.yml
```

See **[docs/architecture.md](docs/architecture.md)** for the full design.

## License

MIT © Ritesh Mamidi
