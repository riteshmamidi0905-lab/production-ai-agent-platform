# ragqa — Retrieval-Augmented Q&A over Your Documents

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small, **dependency-light Retrieval-Augmented Generation (RAG)** engine that
answers questions over a corpus of documents — with a clean library API, a
**FastAPI** microservice, and an optional **LLM generator** hook.

It implements the whole loop from scratch:

```
documents → sentence-aware chunking → TF-IDF index → cosine retrieval → grounded answer (+ citations)
```

The default path is **fully offline**: a from-scratch TF-IDF retriever (no numpy,
no vector DB) plus an **extractive** answerer that returns the best-supported
sentences and cites their source documents. Drop in a `generator(question,
context) -> str` callable to switch to LLM-backed abstractive answers while
keeping the same retrieval and citation scaffolding.

## Why it's built this way

- **No hidden magic.** TF-IDF, cosine similarity, and chunking are implemented
  in readable pure Python, so the retrieval behavior is fully inspectable — the
  point is to *understand* RAG, not import a black box.
- **Grounded + citable by default.** Every answer reports which documents it drew
  from and a confidence score from the top retrieval similarity.
- **Swap-ready.** The index (`add`/`search`) and generator interfaces mirror what
  you'd replace with dense embeddings / FAISS / an LLM in production.

## Install

```bash
git clone https://github.com/riteshmamidi0905-lab/rag-doc-qa.git
cd rag-doc-qa
pip install -e .            # core library is dependency-free
pip install -e ".[dev]"    # pytest + FastAPI test client
```

## Quickstart (library)

```python
from ragqa import RAGPipeline

rag = RAGPipeline(top_k=3)
rag.ingest("return-policy",
           "Returns are accepted within 30 days of purchase. "
           "Items must be unused and in original packaging.")

ans = rag.query("How long do I have to return an item?")
print(ans.answer)      # -> "Returns are accepted within 30 days of purchase. ..."
print(ans.sources)     # -> ["return-policy"]
print(ans.confidence)  # -> top retrieval similarity
```

Run the bundled offline demo:

```bash
python examples/demo.py
```

## Quickstart (API)

```bash
pip install -e ".[api]"
uvicorn ragqa.api:app --reload
```

```bash
curl -X POST localhost:8000/ingest \
     -H 'content-type: application/json' \
     -d '{"doc_id":"faq","text":"Support hours are 9am to 5pm on weekdays."}'

curl -X POST localhost:8000/query \
     -H 'content-type: application/json' \
     -d '{"question":"What are the support hours?"}'
# { "answer": "Support hours are 9am to 5pm on weekdays.",
#   "sources": ["faq"], "confidence": 0.61, "retrieved": [...] }
```

Or with Docker:

```bash
docker build -t ragqa . && docker run -p 8000:8000 ragqa
```

## Plugging in an LLM

```python
from ragqa import RAGPipeline

def generate(question: str, context: str) -> str:
    # call OpenAI / Anthropic / a local model here, passing `context`
    ...

rag = RAGPipeline(generator=generate)
```

Retrieval, chunking, and citations are unchanged — only the final answer step
becomes abstractive.

## Project layout

```
ragqa/
  chunking.py    # sentence-aware, overlapping chunker
  index.py       # from-scratch TF-IDF + cosine search
  retriever.py   # corpus wrapper over the index
  pipeline.py    # ingest → retrieve → answer (extractive or LLM)
  api.py         # FastAPI service (/ingest, /query, /health)
examples/demo.py # runnable offline demo
tests/           # pytest suite (library + API)
```

## Testing

```bash
pytest -q
```

## License

MIT © Ritesh Mamidi
