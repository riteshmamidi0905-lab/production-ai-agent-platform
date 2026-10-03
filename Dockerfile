FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentplatform ./agentplatform
RUN pip install .
COPY corpus ./corpus
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 agent && mkdir -p /app/reports && chown -R agent /app/reports
USER agent
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "agentplatform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
