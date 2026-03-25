"""CleanSlate Streamlit application."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import polars as pl
import requests
import streamlit as st
from streamlit_ace import st_ace

try:
    from annotated_text import annotated_text
except Exception:
    annotated_text = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules import DataProfiler, IntelligentIngestor, SafeExecutor, SuggestionEngine


def set_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 2.3rem; padding-bottom: 2rem;}
            .hero {
                padding: 1rem 1.2rem;
                border: 1px solid rgba(120,120,120,0.2);
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(30,30,30,0.05), rgba(80,80,80,0.08));
                margin-top: 0.8rem;
                margin-bottom: 1rem;
            }
            .muted {opacity: 0.8;}
            div[data-testid="stMetric"] {
                border: 1px solid rgba(120,120,120,0.2);
                border-radius: 10px;
                padding: 0.5rem 0.7rem;
                background: var(--secondary-background-color);
            }
            div[data-testid="stExpander"] {
                border: 1px solid rgba(120,120,120,0.2);
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults: Dict[str, Any] = {
        "df_original": None,
        "df_current": None,
        "profile": None,
        "file_name": None,
        "file_size_mb": 0.0,
        "suggestions": [],
        "selected": {},
        "code_editor": "",
        "llm_provider": "openai",
        "openai_api_key": "",
        "openai_model": "gpt-4o",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def app_header() -> None:
    st.set_page_config(page_title="Aseptic", layout="wide", initial_sidebar_state="expanded")
    set_styles()
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Aseptic</h2>
            <p class="muted" style="margin:0.25rem 0 0 0;">Modern AI-assisted data cleaning workspace</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ollama_status(base_url: str) -> Dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [model.get("name") for model in data.get("models", []) if model.get("name")]
        return {"ok": True, "models": models}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}


def sidebar_settings() -> None:
    with st.sidebar:
        st.subheader("LLM Settings")
        provider = st.selectbox("Provider", ["openai", "ollama"], index=0 if st.session_state.llm_provider == "openai" else 1)
        st.session_state.llm_provider = provider

        if provider == "openai":
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
            st.session_state.openai_model = st.text_input("OpenAI Model", value=st.session_state.openai_model)
        else:
            st.session_state.ollama_base_url = st.text_input("Ollama URL", value=st.session_state.ollama_base_url)
            st.session_state.ollama_model = st.text_input("Ollama Model", value=st.session_state.ollama_model)
            status = ollama_status(st.session_state.ollama_base_url or "http://localhost:11434")
            if status["ok"]:
                if status["models"]:
                    st.success(f"Ollama online ({len(status['models'])} model(s))")
                    st.caption(", ".join(status["models"]))
                else:
                    st.warning("Ollama online, but no models are loaded.")
            else:
                st.error("Ollama not reachable")

        st.markdown("---")
        st.caption("Upload → Profile → Suggest → Execute → Export")


def upload_section() -> None:
    st.subheader("Upload")
    uploaded = st.file_uploader("Drop a file", type=["csv", "json", "jsonl", "xlsx", "parquet"])
    if uploaded is None:
        return

    temp_path = Path(tempfile.gettempdir()) / uploaded.name
    temp_path.write_bytes(uploaded.getbuffer())
    try:
        ingestor = IntelligentIngestor()
        with st.spinner("Loading dataset..."):
            df = ingestor.ingest(str(temp_path))

        st.session_state.df_original = df
        st.session_state.df_current = df
        st.session_state.file_name = uploaded.name
        st.session_state.file_size_mb = uploaded.size / (1024 * 1024)
        st.session_state.profile = None
        st.session_state.suggestions = []
        st.session_state.selected = {}

        cols = st.columns(4)
        cols[0].metric("Rows", f"{df.height:,}")
        cols[1].metric("Columns", df.width)
        cols[2].metric("Size", f"{st.session_state.file_size_mb:.2f} MB")
        cols[3].metric("Format", Path(uploaded.name).suffix.replace(".", "").upper())

        st.dataframe(df.head(10), use_container_width=True)
    finally:
        temp_path.unlink(missing_ok=True)


def profile_section() -> None:
    st.subheader("Profile")
    if st.session_state.df_original is None:
        st.info("Upload a dataset first.")
        return

    if st.button("Generate Profile", type="primary"):
        with st.spinner("Profiling with Polars + health checks..."):
            profiler = DataProfiler()
            st.session_state.profile = profiler.profile_dataframe(st.session_state.df_original)

    profile = st.session_state.profile
    if not profile:
        return

    top = st.columns(3)
    top[0].metric("Total Rows", f"{profile['total_rows']:,}")
    top[1].metric("Total Columns", profile["total_columns"])
    top[2].metric("Health Score", f"{float(profile.get('overall_health_score', 0)):.1f}/100")

    st.markdown("#### Data Health Cards")
    for column, info in profile["columns"].items():
        card = profile.get("health_cards", {}).get(column, {})
        score = float(card.get("health_score", 0))
        status = str(card.get("status", "unknown")).upper()
        with st.expander(f"{column}  •  {status}  •  {score:.1f}/100", expanded=False):
            left, right = st.columns(2)
            left.write(f"**Type:** {info['data_type']}")
            left.write(f"**Nulls:** {info['null_count']} ({info['null_percentage']:.1f}%)")
            right.write(f"**Unique:** {info['unique_count']} ({info['unique_percentage']:.1f}%)")
            if "outliers" in info:
                right.write(f"**IQR Outliers:** {info['outliers'].get('iqr_outliers', 0)}")
            for check in card.get("checks", []):
                icon = "✅" if check.get("passed") else "⚠️"
                st.caption(f"{icon} {check.get('name', 'check')}")


def render_issue(text: str) -> None:
    if annotated_text is not None:
        annotated_text(("Issue", "label"), " ", (text, "detected"))
    else:
        st.markdown(f"**Issue:** {text}")


def suggestion_section() -> None:
    st.subheader("Suggestions")
    if not st.session_state.profile:
        st.info("Generate a profile first.")
        return

    instructions = st.text_area("Optional guidance for the model", placeholder="Focus on date formats, whitespace, and outliers")

    if st.button("Generate Suggestions", type="primary"):
        with st.spinner("Generating cleaning tasks..."):
            engine = SuggestionEngine(
                provider=st.session_state.llm_provider,
                openai_api_key=st.session_state.openai_api_key or None,
                openai_model=st.session_state.openai_model,
                ollama_base_url=st.session_state.ollama_base_url,
                ollama_model=st.session_state.ollama_model,
            )
            st.session_state.suggestions = engine.generate_suggestions(
                st.session_state.profile,
                custom_instructions=instructions or None,
            )
            st.session_state.selected = {f"s_{idx}": True for idx, _ in enumerate(st.session_state.suggestions)}

    if not st.session_state.suggestions:
        return

    st.caption("Accept or reject each task, then edit and execute in the next section.")
    for idx, suggestion in enumerate(st.session_state.suggestions):
        key = f"s_{idx}"
        with st.expander(f"{suggestion.column} • confidence {suggestion.confidence_score:.0%}", expanded=False):
            st.session_state.selected[key] = st.toggle("Accept", value=st.session_state.selected.get(key, True), key=f"toggle_{key}")
            render_issue(suggestion.issue)
            st.markdown(f"**Fix:** {suggestion.fix_description}")
            st.code(suggestion.python_code, language="python")


def build_pipeline_code(suggestions: List[Any], selected: Dict[str, bool]) -> str:
    lines = ["# Review/edit functions below before execution"]
    for idx, suggestion in enumerate(suggestions):
        if selected.get(f"s_{idx}"):
            lines.append("")
            lines.append(f"# Step {idx + 1}: {suggestion.issue}")
            lines.append(suggestion.python_code)
    return "\n".join(lines)


def execute_export_section() -> None:
    st.subheader("Execute & Export")
    if st.session_state.df_original is None:
        st.info("Upload a dataset first.")
        return

    code_default = build_pipeline_code(st.session_state.suggestions, st.session_state.selected)
    edited = st_ace(
        value=code_default,
        language="python",
        theme="tomorrow_night",
        keybinding="vscode",
        min_lines=18,
        max_lines=36,
        auto_update=True,
        wrap=True,
        key="editor",
    )
    st.session_state.code_editor = edited or ""

    col_exec, col_export = st.columns([2, 1])

    with col_exec:
        if st.button("Run Cleaning Pipeline", type="primary"):
            if not st.session_state.code_editor.strip():
                st.error("No code to execute.")
                return
            with st.spinner("Executing restricted pipeline..."):
                executor = SafeExecutor()
                st.session_state.df_current = executor.execute_cleaning_function(
                    st.session_state.df_original,
                    st.session_state.code_editor,
                    function_name="clean_data",
                )
            st.success("Pipeline executed")
            st.dataframe(st.session_state.df_current.head(10), use_container_width=True)

    with col_export:
        fmt = st.selectbox("Export format", ["csv", "parquet", "json"])
        if st.button("Prepare Download"):
            if st.session_state.df_current is None:
                st.error("Run the pipeline first.")
                return
            out = Path(tempfile.gettempdir()) / f"cleanslate_export.{fmt}"
            executor = SafeExecutor()
            executor.export_dataset(st.session_state.df_current, str(out), fmt)
            with open(out, "rb") as file_handle:
                st.download_button(
                    "Download cleaned dataset",
                    data=file_handle.read(),
                    file_name=f"cleaned_dataset.{fmt}",
                    mime="application/octet-stream",
                )


def main() -> None:
    init_state()
    app_header()
    sidebar_settings()

    upload_section()
    st.markdown("---")
    profile_section()
    st.markdown("---")
    suggestion_section()
    st.markdown("---")
    execute_export_section()


if __name__ == "__main__":
    main()
