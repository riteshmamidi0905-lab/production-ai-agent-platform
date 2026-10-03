# Next builds, and the seams already in place

## Project 2 — Real-Time Voice AI Agent
Streaming STT → agent → tools/RAG → streaming TTS with barge-in.

Reuse as-is: the **engine** (plan/tools/RAG/verify), **EventHub + WebSocket** (`/v1/runs/{id}/ws`), **ApprovalBroker** (voice confirmations for consequential tools), **telemetry** (per-stage latency spans), **guardrails**, **conversations** (memory), **eval runner**.

To add:
- `voice/` package: `STTProvider` / `TTSProvider` interfaces mirroring `LLMProvider` (streaming async iterators), a duplex WebSocket endpoint carrying audio frames + control messages, VAD/endpointing, a turn manager.
- **Barge-in**: cancel the in-flight run (`engine.cancel` already propagates `CancelledError` through provider/tool awaits) and flush TTS.
- **Streaming synthesis**: add `stream_text` to providers so TTS can start on the first sentence; keep the structured `Report` for citations (spoken as "according to the lakehouse README").
- **Latency instrumentation**: spans for `stt_final`, `first_token`, `first_audio` (time-to-first-audio is the headline metric); budget per stage.
- **Eval**: word-error-rate on recorded audio fixtures, interruption-handling cases, time-to-first-audio p50/p95.

## Project 3 — MAREF Agent Reliability Platform
Reproducible agent experiments, trace storage, configuration comparison, regression detection over Task Accuracy, Groundedness, Hallucination Resistance, Instruction Adherence, Consistency, Task Completion.

Already here: `eval/metrics.py` (deterministic implementations of five of the six dimensions' building blocks), `eval/runner.py` (datasets → runs → scores), `eval_runs` history table, `run_traces` + append-only `events`, prompt/version hashes recorded per run, provider abstraction (compare configurations by provider/model/prompt version), regression gate.

To add: an `experiments` table (config = provider × model × prompt versions × retrieval params × seed), batch execution with concurrency + cost caps, per-dimension **statistical comparison** (bootstrap CIs, paired tests — the author's `expkit` ports directly), trace diffing UI, optional LLM-judge scorers behind the same metric interface with judge-calibration tracking, and held-out vs dev dataset splits. MAREF itself is a framework in development: this platform would be its measurement substrate, not a claim of results.
