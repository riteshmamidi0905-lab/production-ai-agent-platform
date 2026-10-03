> **▶ [Live demo](https://riteshmamidi0905-lab.github.io/demos/llm-eval-framework.html)** — the interactive evaluation dashboard (scores, per-criterion breakdown, case table) this project generates.

# llmeval — LLM Output Evaluation Platform

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A **rubric-driven platform for evaluating large language model outputs** —
measuring faithfulness, catching hallucinations, checking structured output, and
gating model or prompt changes in CI. It pairs a **deterministic, offline metric
suite** (no API keys needed) with an optional **LLM-as-judge** backend, a **REST
API**, a **CLI**, and a self-contained **interactive HTML dashboard**.

```
cases (JSONL) → Rubric (weighted criteria) → Evaluator → Metrics / LLM judges → JSON · HTML dashboard · CI gate
```

> Built to mirror real GenAI product-operations work: turning fuzzy quality and
> policy guidelines into measurable, weighted criteria and a repeatable pass/fail
> signal — then making the results legible to engineering, policy, and ops.

---

## Highlights

- **14 built-in metrics** — faithfulness, token-F1, ROUGE-L, BLEU, exact/contains
  match, number match, JSON-schema validity, answer relevance, conciseness,
  keyword & regex checks, refusal and safety gates. ([full reference](docs/metrics.md))
- **Weighted rubrics with hard thresholds** — a criterion can carry little weight
  yet still hard-fail a case (safety, no-refusal, minimum faithfulness).
- **Interactive HTML dashboard** — summary cards, per-criterion bar chart, and a
  filterable/sortable per-case table. Self-contained, no CDN, light/dark aware.
- **REST API** (`FastAPI`) and a **CLI** with a `--fail-under` **CI gate**.
- **LLM-as-judge** — score any criterion with a model via a vendor-neutral
  `chat(prompt) -> str` callable (OpenAI/Anthropic adapters included).
- **Benchmark suite** — an 18-case RAG-QA benchmark with a per-category breakdown.
- **Zero core dependencies**, 33 passing tests.

## Install

```bash
git clone https://github.com/riteshmamidi0905-lab/llm-eval-framework.git
cd llm-eval-framework
pip install -e .            # core is dependency-free
pip install -e ".[dev]"    # pytest + FastAPI + pyyaml for tests/API/YAML
```

## Quickstart

```bash
# Run the bundled benchmark → prints a summary + per-category pass rates,
# and writes an interactive dashboard you can open in a browser:
python benchmarks/run_benchmark.py          # -> benchmarks/report.html

# Or evaluate any dataset via the CLI, emitting JSON + an HTML dashboard:
python -m llmeval.cli \
    --rubric benchmarks/rubric.json \
    --cases benchmarks/dataset.jsonl \
    --json report.json --html dashboard.html --fail-under 0.5
```

The dashboard surfaces exactly the cases that matter — a hallucinated "free
parking lot," a wrong price, a spurious refusal — sorted to the top when you
sort by score:

```
MEAN SCORE 65.1%   PASS RATE 61.1%   CASES 18   CRITERIA 8
score by criterion:  safety 1.00 · no_refusal 0.94 · number_accuracy 0.89 · faithfulness 0.67 · … · structure 0.41
parking-halluc  FAIL 0.52   faithfulness 0.10 ✗  (hallucinated a free lot)
warranty-refuse FAIL 0.25   no_refusal 0.00 ✗    (refused a benign question)
```

## Define a rubric (data, not code)

Rubrics are JSON/YAML so policy and quality partners can edit guidelines without
touching the engine:

```json
{
  "name": "rag_qa_comprehensive_v1",
  "criteria": [
    { "name": "faithfulness", "metric": "faithfulness", "weight": 0.30, "threshold": 0.5 },
    { "name": "answer_f1",    "metric": "token_f1",     "weight": 0.20 },
    { "name": "number_accuracy","metric": "number_match","weight": 0.15 },
    { "name": "no_refusal",   "metric": "no_refusal",   "weight": 0.05, "threshold": 1.0 },
    { "name": "safety",       "metric": "safety_flag",  "weight": 0.05, "threshold": 1.0 }
  ]
}
```

Weights normalize automatically; a `threshold` turns a criterion into a hard gate.

## REST API

```bash
pip install -e ".[api]"
uvicorn llmeval.server:app --reload
```

```bash
curl -s localhost:8000/metrics                       # list available metrics
curl -s -X POST localhost:8000/evaluate  -d @payload.json -H 'content-type: application/json'
curl -s -X POST localhost:8000/evaluate/html -d @payload.json -H 'content-type: application/json' > dashboard.html
```

## LLM-as-judge (optional)

```python
from llmeval import Evaluator, Rubric, Criterion
from llmeval.judges import LLMJudge, anthropic_chat

rubric = Rubric("qa", [Criterion("helpfulness", "llm_judge", weight=1.0)])
judge = LLMJudge(chat=anthropic_chat(model="claude-3-5-sonnet-latest"),
                 criterion="helpfulness",
                 guideline="1.0 if the answer fully resolves the question, 0 if not.")
Evaluator(rubric, custom_scorers={"helpfulness": judge}).evaluate(cases)
```

## Add your own metric

```python
from llmeval.metrics import register
from llmeval.cases import EvalCase

@register("ends_with_source")
def ends_with_source(case: EvalCase, **_) -> float:
    return 1.0 if "source:" in case.output.lower() else 0.0
```

## Project layout

```
llmeval/
  rubric.py      # Criterion / Rubric (+ JSON/YAML)
  cases.py       # EvalCase + dataset loaders
  metrics.py     # registry of 14 deterministic scoring functions
  judges.py      # optional LLM-as-judge backends
  evaluator.py   # scoring, weighted aggregation, thresholds, reporting
  report.py      # JSON + interactive self-contained HTML dashboard
  server.py      # FastAPI service (/evaluate, /evaluate/html, /metrics)
  cli.py         # `python -m llmeval.cli ... --html --fail-under`
benchmarks/      # 18-case RAG-QA benchmark + runner (per-category breakdown)
docs/            # architecture + metric reference
examples/  tests/  (33 passing)
```

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/metrics.md`](docs/metrics.md) for details.

## License

MIT © Ritesh Mamidi
