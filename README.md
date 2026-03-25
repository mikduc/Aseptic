# Aseptic

Modern Streamlit workspace for profiling and cleaning data with Polars + LLM guidance.

## Core Stack
- Frontend: Streamlit + `streamlit-ace` (+ optional `st-annotated-text`)
- Data engine: Polars
- Profiling: Polars metrics + Great Expectations-backed health cards (with runtime-safe fallback)
- LLM: OpenAI or Ollama via `SuggestionEngine`
- Execution: `RestrictedPython` sandbox

## Run
```bash
# from project root
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## Configure LLM
Use the sidebar in the app:
- OpenAI: API key + model
- Ollama: base URL + model

The UI also checks Ollama availability and loaded models.

## Execution Behavior
- Cleaning code runs in a restricted sandbox.
- `import` lines in user-edited cleaning code are stripped before execution.
- `pl` (Polars) is pre-provided in the execution context.
- Cleaning execution is time-limited via `MAX_EXECUTION_SECONDS`.

## Project Layout
- `frontend/app.py` – redesigned app UI
- `backend/modules/ingestor.py` – multi-format ingest
- `backend/modules/profiler.py` – profiling + health cards
- `backend/modules/llm_connector.py` – structured cleaning suggestions
- `backend/modules/executor.py` – restricted execution + export
- `config/settings.py` – settings/env access
- `tests/test_integration.py` – integration tests

## Runtime Notes
- On Python 3.14, Great Expectations is conditionally bypassed to avoid known native crashes; health-card fallback remains active.
- If `st-annotated-text` fails to import due dependency mismatch, UI gracefully falls back to plain text issue rendering.
- Ollama availability checks only allow loopback/local URLs for safer defaults.
