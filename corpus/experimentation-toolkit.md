# Experimentation Analysis Toolkit (`expkit`)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)
![Stats](https://img.shields.io/badge/stats-CUPED%20%C2%B7%20power%20%C2%B7%20FDR-8a5cf6)
![License](https://img.shields.io/badge/license-MIT-green)

An **end-to-end A/B experiment analysis toolkit** that takes a product experiment
from **design** (power / sample size / MDE) through **analysis** (frequentist
tests, **CUPED** variance reduction, bootstrap CIs), **segment** breakdowns and
**guardrails**, to a clear **ship / kill / keep-running decision** and a
self-contained **HTML scorecard**.

This is the work a Product / Data Analyst actually owns: not a dashboard, but the
statistical machinery behind *"do we ship this or not?"*

```
design (power / n / MDE) → analyze (z-test, Welch t, CUPED, bootstrap)
    → segments + guardrails → FDR correction → SHIP / KILL / INCONCLUSIVE + scorecard
```

> Applies the statistics, regression, and experimentation methods from the
> **Google Advanced Data Analytics** and **MS Business Analytics** coursework,
> built with software-engineering rigor (typed API, 11 tests) from an
> **M.Tech in Software Engineering**.

---

## The scenario

A team ships a **checkout redesign** to half of new users. We need to know whether
it actually lifts **conversion** (primary), what it does to **revenue per user**
(secondary), and whether it hurts **engagement** (a guardrail that must not
regress) — overall and by device.

The toolkit generates a realistic, seeded experiment (24k users, a pre-period
covariate, heterogeneous effects by device) and analyzes it. Result:

| Metric | Control | Treatment | Rel. lift | p-value | |
| --- | --- | --- | --- | --- | --- |
| Conversion rate | 10.93% | 13.43% | **+22.9%** | <0.001 | primary |
| Revenue / user | $2.55 | $3.26 | **+28.0%** | <0.001 | CUPED |
| Session minutes | 20.98 | 20.85 | −0.65% | 0.097 | guardrail ✓ |

**Decision: SHIP** — primary up and significant, no guardrail regression. Effect
is real across all devices but strongest on iOS (+25.6%) vs web (+24.4%).

![scorecard](docs/scorecard.png)

## What it does

- **Design** — `sample_size_proportion`, `power_proportion`, `mde_proportion`:
  size a test *before* running it and post-hoc power *after*.
- **Frequentist tests** — pooled two-proportion **z-test** for rates, **Welch's
  t-test** for continuous metrics, and a **percentile bootstrap** CI for skewed
  metrics like revenue-per-user.
- **CUPED variance reduction** — uses a pre-experiment covariate
  (`prior_28d_sessions`) to strip out pre-existing user differences, tightening
  the revenue CI and buying power without more traffic.
- **Guardrails** — metrics that must not regress; the decision flags any that do.
- **Segments** — heterogeneous treatment effects by device (iOS vs Android vs web).
- **Multiple-comparison control** — Benjamini–Hochberg (FDR) and Bonferroni, so
  secondary metrics don't manufacture false positives.
- **Decision + scorecard** — an auditable SHIP / KILL / INCONCLUSIVE call and a
  polished, dependency-free HTML scorecard.

## Quickstart

```bash
git clone https://github.com/riteshmamidi0905-lab/experimentation-toolkit.git
cd experimentation-toolkit
pip install -e ".[dev]"

python examples/generate_and_run.py       # full design → analyze → decide demo
pytest -q                                  # 11 passing tests
```

CLI:

```bash
# Design: users/group to detect a +1.5pp lift on an 11% baseline at 80% power
python -m expkit.cli design --baseline 0.11 --mde 0.015 --power 0.8
#  → 6,950 users/group

# Analyze a results CSV and write a scorecard
python -m expkit.cli analyze --data data/experiment.csv \
    --primary converted:binary \
    --secondary "revenue:continuous:prior_28d_sessions" \
    --guardrail session_minutes:continuous \
    --segment device --scorecard scorecard.html
```

## Methodology notes

**Why CUPED?** Revenue per user is noisy and heavy-tailed, so a plain t-test needs
a lot of traffic. CUPED regresses out a pre-period covariate `X`:
`Y_adj = Y − θ·(X − X̄)`, with `θ = Cov(Y,X)/Var(X)`. The expected treatment effect
is unchanged but variance drops, so the same data yields a tighter CI and higher
power. The test suite verifies the effect estimate is preserved while the CI
shrinks.

**Why guardrails and FDR?** Shipping on a single significant metric is how teams
ship regressions. Guardrails catch the "won the primary, hurt engagement" case;
Benjamini–Hochberg keeps the family-wise error in check once you're looking at
several metrics at once.

**Decision framework.** `SHIP` when the primary moves significantly in the good
direction with no guardrail regression; `KILL` when it moves the wrong way;
`INCONCLUSIVE` otherwise (with a note on whether the test was powered).

## Project layout

```
expkit/
  stats.py        # z-test, Welch t, bootstrap, CUPED, power/size/MDE, BH/Bonferroni
  experiment.py   # Experiment orchestrator: metrics, segments, ship/kill decision
  report.py       # self-contained HTML scorecard
  data.py         # seeded synthetic experiment generator
  cli.py          # `expkit design ...` / `expkit analyze ...`
examples/generate_and_run.py
tests/            # 11 passing tests (stats correctness + end-to-end)
```

## License

MIT © Ritesh Mamidi
