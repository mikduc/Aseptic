# CleanSlate

Modern Streamlit app for profiling and cleaning datasets with Polars + LLM guidance.

## Stack
- Frontend: Streamlit + streamlit-ace (+ optional st-annotated-text)
- Data engine: Polars
- Profiling: Polars stats + Great Expectations-style health cards (with safe fallback)
- LLM: OpenAI or Ollama via SuggestionEngine
- Execution: RestrictedPython sandbox

## Run
```bash
# from project root
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## Configure LLM
Use the app sidebar:
- OpenAI: API key + model
- Ollama: base URL + model

## Project Structure
- `frontend/app.py` – modern single-page UI workspace
- `backend/modules/ingestor.py` – multi-format ingest
- `backend/modules/profiler.py` – profiling + health cards
- `backend/modules/llm_connector.py` – strict suggestion schema handling
- `backend/modules/executor.py` – restricted execution + export
- `config/settings.py` – env/settings
- `tests/test_integration.py` – integration tests

## Notes
- On Python 3.14, Great Expectations is conditionally disabled at runtime to avoid known native crashes; health-card fallback remains active.
