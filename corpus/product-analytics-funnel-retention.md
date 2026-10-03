# Product Analytics — Funnel, Retention & Cohorts (`prodanalytics`)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
![Focus](https://img.shields.io/badge/analysis-funnel%20%C2%B7%20retention%20%C2%B7%20cohorts-8a5cf6)
![License](https://img.shields.io/badge/license-MIT-green)

The **product-analytics core a growth team lives on**: turn a raw signup + activity
log into an **acquisition funnel**, **retention curves by channel**, a **cohort
retention heatmap**, and a **channel-LTV comparison** — then roll it up into a
self-contained **HTML growth dashboard**.

```
events (signups + weekly activity) → funnel · retention · cohorts · LTV → growth dashboard
```

> Built around the product-analytics and statistics methods from the
> **Google Advanced Data Analytics** and **MS Business Analytics** programs, with
> clean, tested code (6 tests) reflecting an **M.Tech in Software Engineering**.

![dashboard](docs/dashboard.png)

## What the analysis surfaces (from the seeded demo)

- **Funnel:** 8,000 visits → 3,155 signups (39%) → 1,689 activated (54% step) →
  464 subscribed (27% step). The biggest drop is visit→signup.
- **Retention:** week-1 87% decaying to ~40% by week 7 — and the split by channel
  is the real story: **referral** holds ~58% at week 7 while **paid** falls to
  ~20%.
- **Cohorts:** a stable retention shape across ten signup cohorts (no cohort is
  quietly falling apart), shown as a heatmap.
- **Channel LTV proxy:** referral users are worth **~$18** each vs **~$4** for
  paid — a clear signal for where acquisition spend should go.

## What it does

- **Funnel analysis** — overall and step-to-step conversion, optionally segmented
  by channel/country.
- **Retention** — week-N retention curves overall and by acquisition channel.
- **Cohort analysis** — a signup-cohort × weeks-since-signup retention matrix.
- **Channel LTV proxy** — subscribe-rate × ARPU × horizon, a first-order LTV for
  comparing acquisition sources.
- **Dashboard** — a dependency-free HTML page with KPI cards, embedded charts,
  and tables; light/dark aware.

## Quickstart

```bash
git clone https://github.com/riteshmamidi0905-lab/product-analytics-funnel-retention.git
cd product-analytics-funnel-retention
pip install -e ".[dev]"

python examples/run.py     # prints the analyses and writes data/dashboard.html
pytest -q                  # 6 passing tests
```

Use it as a library:

```python
from prodanalytics import generate, funnel_summary, retention_curve, build_dashboard

users, funnel, activity = generate(n_users=8000, seed=7)
print(funnel_summary(funnel))                 # funnel table
print(retention_curve(activity, by="channel"))# retention by channel
build_dashboard(users, funnel, activity, "dashboard.html")
```

## Method notes

- **Cohort retention** is computed as, for each signup-week cohort, the share of
  its members active in each subsequent week (week 0 = 100% by construction).
- **Retention curves** average week-N retention across cohorts; splitting by
  channel exposes that paid traffic converts *and* churns worse — the kind of
  finding that changes budget allocation.
- The **LTV proxy** is deliberately simple (not a survival model) — the honest
  first pass a growth analyst ships before investing in a full LTV model.

## Project layout

```
prodanalytics/
  data.py        # seeded signup + weekly-activity generator
  analysis.py    # funnel, retention curves, cohort matrix, channel LTV
  viz.py         # matplotlib charts (funnel, curves, cohort heatmap)
  dashboard.py   # self-contained HTML dashboard (embedded charts + tables)
examples/run.py
tests/           # 6 passing tests
```

## License

MIT © Ritesh Mamidi
