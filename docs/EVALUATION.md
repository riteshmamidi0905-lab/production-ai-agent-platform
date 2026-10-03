# Evaluation

`python -m agentplatform.cli --embedded-pg eval --repeats 2 [--gate] [--update-baseline]` (or the UI's Evaluation tab, or CI).

**Dataset** `agentplatform/eval/datasets/research_v1.jsonl`: 15 labelled cases over the bundled corpus (READMEs of the author's own repos): 12 single/multi-document questions, 1 calculation, 1 unanswerable (must abstain), 1 prompt-injection (must be blocked). Each case lists `expected_docs` and `required_facts` (alternatives per fact).

| Metric | Definition (all deterministic, from recorded artefacts) |
|---|---|
| groundedness | share of claims supported by their *cited* evidence: ≥ 0.8 content-token coverage **and** every number in the claim appears in that evidence |
| citation F1 | precision/recall of cited documents vs labelled `expected_docs` |
| task completion | share of required facts present in the answer (abstain/blocked cases score expectation met) |
| tool success | ok tool calls / all tool calls |
| instruction adherence | citations present, `max_words` respected, no forbidden phrases |
| consistency | mean pairwise similarity (cited-chunk Jaccard + fact-vector agreement) across repeated runs |
| unsupported numbers | numbers in the answer absent from all evidence |
| latency p50/p95 | end-to-end run latency |

**Regression gate**: no tracked metric may fall > 0.02 below `baselines/<provider>.json`; p95 latency has an absolute cap. A test deliberately breaks retrieval and asserts the gate fails.

## Current baseline (offline provider)
groundedness 1.000 · citation F1 0.882 (precision 0.815, recall 1.0) · task completion 0.967 · tool success 1.0 · adherence 1.0 · unsupported numbers 0.0 · consistency 1.0 · p50 5.6 ms · p95 10.8 ms · 15 cases (14 complete; the injection case fails by design).

## What these numbers mean
- The offline provider is deterministic and extractive: groundedness and consistency are near-ceiling **by construction**. Retrieval-dependent metrics (citation F1, task completion) are the real signal.
- Evaluation improved the system: the first run exposed list items being merged into one giant "sentence" (facts buried), boilerplate numbers in the summary template, a word budget that ignored formatting, and a calculator extractor that dropped outer parentheses. All were fixed in the platform (task completion 0.80 → 0.97, unsupported numbers 0.20 → 0.0). Because the dataset guided those fixes it is a **development set**; add a held-out set before quoting numbers externally.
- One known miss is left visible: `metric-count` (the answer does not enumerate metric names).
- To evaluate a real model: set a provider key and run with `--provider openai|ollama|anthropic`; `--update-baseline` stores that provider's baseline. Real-model runs will make consistency and groundedness informative.
