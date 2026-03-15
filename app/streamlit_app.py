"""
Parkrun Survey Analysis
========================
Streamlit application for analysing SurveyMonkey survey data using LLMs.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/ regardless of working directory
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_engine import AnalysisEngine
from src.llm_interface import PROVIDER_LABELS, get_provider
from src.survey_parser import QuestionType, SurveyParser

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Parkrun Survey Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — parkrun green colour scheme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Primary accent colour */
    :root { --parkrun-green: #006633; --parkrun-light: #e8f5ee; }

    /* Sidebar background */
    [data-testid="stSidebar"] { background-color: #f0f7f3; }

    /* Status badges */
    .status-ok  { color: #006633; font-weight: bold; }
    .status-warn { color: #b35c00; font-weight: bold; }
    .status-err  { color: #c0392b; font-weight: bold; }

    /* Question card */
    .question-card {
        background: #f9f9f9;
        border-left: 4px solid #006633;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }

    /* Keyword chip */
    .kw-chip {
        display: inline-block;
        background: #e8f5ee;
        color: #004d22;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.85rem;
    }

    /* Metric label */
    .metric-label { font-size: 0.8rem; color: #666; margin-bottom: 0; }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #006633; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config loaders — cached so YAML files are only read once
# ---------------------------------------------------------------------------

@st.cache_data
def load_settings() -> dict:
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_prompts() -> dict:
    path = PROJECT_ROOT / "config" / "prompts.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_surveys_dir() -> Path:
    settings = st.session_state.get("settings", {})
    rel = settings.get("app", {}).get("surveys_dir", "data/surveys")
    d = PROJECT_ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "settings": load_settings(),
        "prompts": load_prompts(),
        "survey_data": None,          # Parsed SurveyData object
        "current_survey_name": None,  # Filename of the loaded survey
        "analysis_results": {},       # {stem: {stats, keywords, llm_text}}
        "provider": None,             # Active LLMProvider instance
        "api_key": "",                # Never written to disk
        "provider_name": "",          # "groq" | "openai" | "anthropic"
        "model_name": "",
        "chat_history": [],           # [{role, content}]
        "chat_engine": None,          # AnalysisEngine used for chat
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🏃 Parkrun Survey\nAnalysis Tool")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Home", "📤 Upload Surveys", "📊 Analyse Survey", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Status**")

        # Survey loaded
        if st.session_state.current_survey_name:
            st.markdown(f'<span class="status-ok">✓ Survey loaded</span>', unsafe_allow_html=True)
            st.caption(st.session_state.current_survey_name)
        else:
            st.markdown('<span class="status-warn">○ No survey loaded</span>', unsafe_allow_html=True)

        # LLM configured
        if st.session_state.provider:
            label = PROVIDER_LABELS.get(st.session_state.provider_name, st.session_state.provider_name)
            st.markdown(f'<span class="status-ok">✓ LLM: {label}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">○ No LLM configured</span>', unsafe_allow_html=True)

        # Analysis run
        if st.session_state.analysis_results:
            n = len(st.session_state.analysis_results)
            st.markdown(f'<span class="status-ok">✓ {n} questions analysed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">○ No analysis yet</span>', unsafe_allow_html=True)

        st.divider()
        st.caption("v" + st.session_state.settings.get("app", {}).get("version", "0.1.0"))

    return page


# ===========================================================================
# PAGE: Home
# ===========================================================================

def page_home() -> None:
    st.title("🏃 Parkrun Survey Analysis")
    st.markdown(
        "**AI-powered insights from parkrun community surveys**  \n"
        "Built by a Digital Ambassador volunteer to help parkrun understand participant "
        "and volunteer feedback at scale."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What this tool does")
        st.markdown("""
- Parses **SurveyMonkey** CSV and XLSX exports automatically
- Extracts **keywords** and **basic statistics** from every question
- Uses an **LLM** (AI) to identify themes, sentiment, and actionable insights
- Provides a **chat interface** so you can ask questions about your survey data
        """)

    with col2:
        st.subheader("How AI is used")
        st.markdown("""
The tool sends a small sample of anonymised survey responses to your chosen
AI provider (Groq, OpenAI, or Anthropic) along with a parkrun-specific prompt.
The AI then returns structured insights about themes, sentiment, and recommendations.

**Your API key is never saved to disk.** It lives only in your browser session
and disappears when you close the tab.

Prompts are stored in `config/prompts.yaml` and can be edited to customise
the analysis for your specific survey.
        """)

    st.divider()
    st.subheader("Quick Start")

    steps = [
        ("⚙️ **Settings**", "Add your API key for Groq (free), OpenAI, or Anthropic. "
                             "See `docs/llm_setup.md` for step-by-step instructions."),
        ("📤 **Upload Surveys**", "Upload your SurveyMonkey CSV or XLSX export. "
                                   "The file is saved to `data/surveys/` in this repository."),
        ("📊 **Analyse Survey**", "Select your survey, click **Run Analysis**, and explore "
                                   "the results. Use the chat at the bottom to ask questions."),
    ]

    for icon_title, desc in steps:
        with st.container(border=True):
            st.markdown(f"{icon_title}  \n{desc}")

    st.divider()
    st.subheader("About parkrun")
    st.info(
        "parkrun is a free, community event where you can walk, jog, run, volunteer or spectate. "
        "parkrun is 5k and takes place every Saturday morning. junior parkrun is 2k, dedicated to "
        "4-14 year olds and their families, every Sunday morning. parkrun is positive, welcoming "
        "and inclusive, there is no time limit and no one finishes last. Everyone is welcome to come along."
    )

    st.markdown(
        "📖 [LLM Setup Guide](https://github.com/olisimmonds/parkrun_survey_analysis/blob/main/docs/llm_setup.md) &nbsp;·&nbsp; "
        "💻 [GitHub Repository](https://github.com/olisimmonds/parkrun_survey_analysis)",
        unsafe_allow_html=False,
    )


# ===========================================================================
# PAGE: Upload Surveys
# ===========================================================================

def page_upload() -> None:
    st.title("📤 Upload Surveys")
    st.markdown(
        "Upload SurveyMonkey exports here. Files are saved to `data/surveys/` "
        "in this repository and will persist between sessions."
    )

    surveys_dir = get_surveys_dir()

    # ---- Upload widget ----
    st.subheader("Upload a new survey")
    uploaded = st.file_uploader(
        "Choose a CSV or XLSX file exported from SurveyMonkey",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        help="Go to SurveyMonkey → Analyse Results → Export → Export All → Individual Responses",
    )

    if uploaded is not None:
        save_path = surveys_dir / uploaded.name
        if save_path.exists():
            st.warning(f"**{uploaded.name}** already exists. It will be overwritten.")

        if st.button("💾 Save survey", type="primary"):
            with st.spinner("Saving and parsing survey..."):
                raw_bytes = uploaded.read()
                save_path.write_bytes(raw_bytes)

                # Quick parse to validate and show preview
                try:
                    parser = SurveyParser(
                        metadata_columns=st.session_state.settings
                        .get("parser", {})
                        .get("metadata_columns")
                    )
                    survey = parser.parse(raw_bytes, filename=uploaded.name)
                    st.success(
                        f"✅ **{uploaded.name}** saved successfully!  \n"
                        f"Detected **{len(survey.questions)} questions** across "
                        f"**{survey.n_respondents} respondents**."
                    )
                    _show_survey_preview(survey)
                except Exception as e:
                    st.error(f"File saved but could not be parsed: {e}")

    st.divider()

    # ---- List uploaded surveys ----
    st.subheader("Uploaded surveys")
    survey_files = sorted(
        list(surveys_dir.glob("*.csv")) + list(surveys_dir.glob("*.xlsx"))
    )

    if not survey_files:
        st.info("No surveys uploaded yet. Use the uploader above to add your first survey.")
        return

    for f in survey_files:
        col_name, col_size, col_action = st.columns([4, 1, 1])
        with col_name:
            icon = "📊" if f.suffix == ".csv" else "📗"
            st.markdown(f"{icon} **{f.name}**")
        with col_size:
            size_kb = f.stat().st_size / 1024
            st.caption(f"{size_kb:.1f} KB")
        with col_action:
            if st.button("🗑️ Delete", key=f"del_{f.name}"):
                f.unlink()
                # Clear analysis if this was the active survey
                if st.session_state.current_survey_name == f.name:
                    st.session_state.current_survey_name = None
                    st.session_state.survey_data = None
                    st.session_state.analysis_results = {}
                    st.session_state.chat_history = []
                    st.session_state.chat_engine = None
                st.rerun()


def _show_survey_preview(survey) -> None:
    """Show a compact table of detected questions."""
    st.subheader("Detected questions")
    rows = []
    for q in survey.questions:
        rows.append({
            "Question": q.stem[:80] + ("…" if len(q.stem) > 80 else ""),
            "Type": q.question_type.label,
            "Responses": q.n_answered,
            "Response rate": f"{round(q.n_answered / q.n_total * 100)}%" if q.n_total else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Analyse Survey
# ===========================================================================

def page_analyse() -> None:
    st.title("📊 Analyse Survey")

    surveys_dir = get_surveys_dir()
    survey_files = sorted(
        list(surveys_dir.glob("*.csv")) + list(surveys_dir.glob("*.xlsx"))
    )

    if not survey_files:
        st.warning(
            "No surveys found. Go to **Upload Surveys** to add a SurveyMonkey export first."
        )
        return

    # ---- Survey selector ----
    file_names = [f.name for f in survey_files]
    current_idx = 0
    if st.session_state.current_survey_name in file_names:
        current_idx = file_names.index(st.session_state.current_survey_name)

    selected_name = st.selectbox(
        "Select a survey",
        options=file_names,
        index=current_idx,
    )

    # Load survey data if selection changed or not yet loaded
    if selected_name != st.session_state.current_survey_name or st.session_state.survey_data is None:
        with st.spinner(f"Parsing {selected_name}..."):
            try:
                parser = SurveyParser(
                    metadata_columns=st.session_state.settings
                    .get("parser", {})
                    .get("metadata_columns")
                )
                survey = parser.parse(surveys_dir / selected_name, filename=selected_name)
                st.session_state.survey_data = survey
                st.session_state.current_survey_name = selected_name
                # Clear old analysis when switching surveys
                st.session_state.analysis_results = {}
                st.session_state.chat_history = []
                st.session_state.chat_engine = None
            except Exception as e:
                st.error(f"Could not parse survey: {e}")
                return

    survey = st.session_state.survey_data
    if survey is None:
        return

    # ---- Survey summary ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Respondents", survey.n_respondents)
    col2.metric("Questions", len(survey.questions))
    answered_qs = sum(1 for q in survey.questions if q.n_answered > 0)
    col3.metric("Questions with responses", answered_qs)

    st.divider()

    # ---- Question selection ----
    with st.expander("⚙️ Select questions to analyse", expanded=False):
        st.caption("All questions are selected by default. Deselect any you want to skip.")
        select_all = st.checkbox("Select / deselect all", value=True, key="select_all")
        question_selections = {}
        for i, q in enumerate(survey.questions):
            default = select_all
            label = f"{q.question_type.label}  |  {q.stem[:70]}{'…' if len(q.stem) > 70 else ''}"
            question_selections[q.stem] = st.checkbox(label, value=default, key=f"qsel_{i}")

    selected_questions = [q for q in survey.questions if question_selections.get(q.stem, True)]

    # ---- Action buttons ----
    col_basic, col_llm, _ = st.columns([2, 2, 4])

    run_basic = col_basic.button("📈 Run Basic Stats", type="secondary")
    has_provider = st.session_state.provider is not None
    run_llm = col_llm.button(
        "🤖 Run Full Analysis",
        type="primary",
        disabled=not has_provider,
        help="Configure an LLM provider in Settings first." if not has_provider else None,
    )

    if not has_provider:
        st.caption("💡 Add an API key in **Settings** to enable LLM analysis.")

    # ---- Run basic stats ----
    if run_basic:
        _run_basic_stats(selected_questions, survey)

    # ---- Run full LLM analysis ----
    if run_llm:
        _run_llm_analysis(selected_questions, survey)

    # ---- Display results ----
    if st.session_state.analysis_results:
        _display_results(survey)

    # ---- Chat interface ----
    if st.session_state.analysis_results:
        _render_chat()


def _run_basic_stats(questions, survey) -> None:
    """Compute basic stats for all selected questions without LLM."""
    from src.keyword_extraction import KeywordExtractor

    settings = st.session_state.settings
    custom_stops = settings.get("parser", {}).get("custom_stopwords", [])
    extractor = KeywordExtractor(custom_stopwords=custom_stops)

    engine = AnalysisEngine(
        survey_data=survey,
        settings=settings,
        prompts=st.session_state.prompts,
        provider=None,
    )

    progress = st.progress(0, text="Computing statistics…")
    results = dict(st.session_state.analysis_results)  # preserve existing LLM results

    for i, q in enumerate(questions):
        stats = engine.get_basic_stats(q)
        keywords = engine.get_keywords(q)

        if q.stem not in results:
            results[q.stem] = {}
        results[q.stem]["stats"] = stats
        results[q.stem]["keywords"] = keywords
        results[q.stem]["stem"] = q.stem
        results[q.stem]["question"] = q

        progress.progress((i + 1) / len(questions), text=f"Processing: {q.stem[:50]}…")

    progress.empty()
    st.session_state.analysis_results = results

    # Build chat context from stats + keywords so chat works even without LLM analysis
    analyses = [
        {
            "stem": v.get("stem", ""),
            "question": v.get("question"),
            "stats": v.get("stats", {}),
            "keywords": v.get("keywords", []),
            "llm_text": v.get("llm_text", ""),
        }
        for v in results.values()
    ]
    engine.build_chat_context(analyses)
    st.session_state.chat_engine = engine

    st.success(f"✅ Basic stats computed for {len(questions)} questions.")
    st.rerun()


def _run_llm_analysis(questions, survey) -> None:
    """Run full LLM analysis for all selected questions."""
    settings = st.session_state.settings
    engine = AnalysisEngine(
        survey_data=survey,
        settings=settings,
        prompts=st.session_state.prompts,
        provider=st.session_state.provider,
    )

    results = dict(st.session_state.analysis_results)
    completed_analyses = []

    progress = st.progress(0, text="Starting analysis…")
    status_text = st.empty()

    for i, q in enumerate(questions):
        status_text.markdown(
            f"**Analysing question {i + 1} of {len(questions)}:**  \n"
            f"`{q.stem[:80]}`"
        )

        # Basic stats first
        stats = engine.get_basic_stats(q)
        keywords = engine.get_keywords(q)

        if q.stem not in results:
            results[q.stem] = {}
        results[q.stem]["stats"] = stats
        results[q.stem]["keywords"] = keywords
        results[q.stem]["stem"] = q.stem
        results[q.stem]["question"] = q

        # LLM analysis — collect chunks silently
        chunks = []
        try:
            for chunk in engine.stream_llm_analysis(q, stats, keywords):
                chunks.append(chunk)
        except Exception as e:
            chunks = [f"⚠️ Error during LLM analysis: {e}"]

        llm_text = "".join(chunks)
        results[q.stem]["llm_text"] = llm_text

        completed_analyses.append({
            "stem": q.stem,
            "question": q,
            "stats": stats,
            "keywords": keywords,
            "llm_text": llm_text,
        })

        progress.progress((i + 1) / len(questions))

    progress.empty()
    status_text.empty()
    st.session_state.analysis_results = results

    # Build chat context
    engine.build_chat_context(completed_analyses)
    st.session_state.chat_engine = engine

    st.success(f"✅ Full analysis complete for {len(questions)} questions.")
    st.rerun()


def _display_results(survey) -> None:
    """Render analysis results for each question."""
    st.divider()
    st.subheader("Results")
    results = st.session_state.analysis_results

    # Only show questions that have results
    question_map = {q.stem: q for q in survey.questions}
    stems_with_results = [
        q.stem for q in survey.questions if q.stem in results
    ]

    for stem in stems_with_results:
        r = results[stem]
        q = question_map.get(stem)
        if q is None:
            continue

        with st.expander(f"**{stem[:100]}**  `{q.question_type.label}`", expanded=False):
            _render_question_result(q, r)


def _render_question_result(q, r: dict) -> None:
    """Render the full result panel for one question."""
    stats = r.get("stats", {})
    keywords = r.get("keywords", [])
    llm_text = r.get("llm_text", "")

    tabs = ["📈 Stats & Keywords"]
    if llm_text:
        tabs.append("🤖 LLM Analysis")

    tab_objects = st.tabs(tabs)

    # ---- Stats & Keywords tab ----
    with tab_objects[0]:
        q_type = q.question_type

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Response rate**")
            st.metric("Answered", f"{stats.get('n_answered', 0):,}")
            st.metric("Total", f"{stats.get('n_total', 0):,}")
            pct = stats.get("answered_pct", 0)
            st.progress(pct / 100, text=f"{pct}% response rate")

        with col2:
            if q_type == QuestionType.FREE_TEXT:
                st.markdown("**Text stats**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg words", stats.get("avg_words", "—"))
                c2.metric("Median words", stats.get("median_words", "—"))
                c3.metric("Total words", f"{stats.get('total_words', 0):,}")

            elif q_type == QuestionType.RATING:
                mean = stats.get("mean")
                if mean is not None:
                    st.markdown("**Rating stats**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean", f"{mean:.2f}")
                    c2.metric("Median", stats.get("median", "—"))
                    c3.metric("Std dev", f"{stats.get('std', 0):.2f}")

                    dist = stats.get("distribution", {})
                    if dist:
                        dist_df = pd.DataFrame(
                            {"Score": list(dist.keys()), "Count": list(dist.values())}
                        ).set_index("Score")
                        st.bar_chart(dist_df)

            elif q_type in (QuestionType.SINGLE_CHOICE, QuestionType.MULTI_SELECT):
                vc = stats.get("value_counts", {})
                pcts = stats.get("value_pcts", {})
                if vc:
                    st.markdown("**Response breakdown**")
                    rows = [
                        {"Option": k, "Count": v, "Percentage": f"{pcts.get(k, 0):.1f}%"}
                        for k, v in list(vc.items())[:15]
                    ]
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        # Keywords
        if keywords:
            st.markdown("**Keywords & phrases**")
            chips = "".join(
                f'<span class="kw-chip">{word}</span>'
                for word, _ in keywords[:20]
            )
            st.markdown(chips, unsafe_allow_html=True)

    # ---- LLM Analysis tab ----
    if llm_text and len(tab_objects) > 1:
        with tab_objects[1]:
            st.markdown(llm_text)


# ===========================================================================
# Chat interface
# ===========================================================================

def _render_chat() -> None:
    st.divider()
    st.subheader("💬 Chat with your survey data")
    st.caption(
        "Ask questions about the survey results. The AI will use the analysis "
        "above to answer. Requires a configured LLM provider."
    )

    if not st.session_state.provider:
        st.warning("Add an API key in **Settings** to use the chat feature.")
        return

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Example prompts for first-time users
    if not st.session_state.chat_history:
        st.markdown("**Example questions:**")
        example_cols = st.columns(3)
        examples = [
            "What themes appear most frequently?",
            "What are participants most positive about?",
            "Are there any negative sentiments?",
        ]
        for col, example in zip(example_cols, examples):
            if col.button(example, key=f"ex_{example[:20]}"):
                _handle_chat_message(example)
                st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask a question about the survey results…"):
        _handle_chat_message(user_input)
        st.rerun()


def _handle_chat_message(user_input: str) -> None:
    """Process a chat message and stream the response."""
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get or create engine
    engine = st.session_state.chat_engine
    if engine is None:
        # Create engine without pre-built context — will use all available results
        survey = st.session_state.survey_data
        engine = AnalysisEngine(
            survey_data=survey,
            settings=st.session_state.settings,
            prompts=st.session_state.prompts,
            provider=st.session_state.provider,
        )
        # Build context from existing results — include stats and question objects
        # so the chat has real data even if no LLM analysis was run
        analyses = [
            {
                "stem": v.get("stem", ""),
                "question": v.get("question"),
                "stats": v.get("stats", {}),
                "keywords": v.get("keywords", []),
                "llm_text": v.get("llm_text", ""),
            }
            for v in st.session_state.analysis_results.values()
        ]
        engine.build_chat_context(analyses)
        st.session_state.chat_engine = engine

    # Stream response
    history_for_llm = st.session_state.chat_history[:-1]  # exclude last user msg

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in engine.answer_chat(user_input, history=history_for_llm):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})


# ===========================================================================
# PAGE: Settings
# ===========================================================================

def page_settings() -> None:
    st.title("⚙️ Settings")

    st.info(
        "**Privacy note:** API keys are stored in your browser session only. "
        "They are never written to disk or sent anywhere other than your chosen LLM provider."
    )

    settings = st.session_state.settings

    # ---- Provider selection ----
    st.subheader("LLM Provider")

    provider_options = ["groq", "openai", "anthropic"]
    provider_labels = [PROVIDER_LABELS[p] for p in provider_options]

    current_provider = st.session_state.provider_name or settings.get("llm", {}).get("default_provider", "groq")
    try:
        current_idx = provider_options.index(current_provider)
    except ValueError:
        current_idx = 0

    selected_label = st.selectbox(
        "Choose your LLM provider",
        options=provider_labels,
        index=current_idx,
        help="Groq is free. OpenAI and Anthropic require a paid account (low cost).",
    )
    selected_provider = provider_options[provider_labels.index(selected_label)]

    # Provider info
    provider_info = {
        "groq": (
            "**Groq (Recommended for getting started)**  \n"
            "Free tier available — no credit card required. "
            "Uses open-source Llama models hosted by Groq.  \n"
            "📖 See [docs/llm_setup.md](docs/llm_setup.md) for setup instructions."
        ),
        "openai": (
            "**OpenAI**  \n"
            "Requires a paid account. gpt-4o-mini is very affordable "
            "(approximately £0.001–0.01 per survey analysis).  \n"
            "📖 See [docs/llm_setup.md](docs/llm_setup.md) for setup instructions."
        ),
        "anthropic": (
            "**Anthropic**  \n"
            "Requires a paid account. claude-haiku is the most affordable option.  \n"
            "📖 See [docs/llm_setup.md](docs/llm_setup.md) for setup instructions."
        ),
    }
    st.markdown(provider_info[selected_provider])

    # ---- API key ----
    st.subheader("API Key")
    current_key = st.session_state.api_key or ""
    api_key = st.text_input(
        f"{PROVIDER_LABELS[selected_provider]} API Key",
        value=current_key,
        type="password",
        placeholder="Paste your API key here…",
        help="Your key is stored in memory only and disappears when you close the browser tab.",
    )

    # ---- Model selection ----
    st.subheader("Model")
    provider_settings = settings.get("llm", {}).get("providers", {}).get(selected_provider, {})
    default_model = provider_settings.get("default_model", "")

    # Common model options per provider
    model_options = {
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-haiku-20240307",
        ],
    }

    options = model_options.get(selected_provider, [default_model])
    current_model = st.session_state.model_name or default_model
    if current_model not in options:
        options = [current_model] + options if current_model else options

    model = st.selectbox("Model", options=options)

    # ---- Advanced ----
    with st.expander("Advanced settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0,
            value=float(provider_settings.get("temperature", 0.3)),
            step=0.05,
            help="Lower = more consistent, higher = more creative.",
        )
        max_tokens = st.number_input(
            "Max tokens",
            min_value=200, max_value=8000,
            value=int(provider_settings.get("max_tokens", 2000)),
            step=100,
            help="Maximum length of each LLM response.",
        )
        if selected_provider == "groq":
            rate_delay = st.slider(
                "Rate limit delay (seconds between calls)",
                min_value=0.0, max_value=10.0,
                value=float(provider_settings.get("rate_limit_delay_seconds", 2.0)),
                step=0.5,
                help="Increase if you see rate limit errors from Groq.",
            )
        else:
            rate_delay = 0.0

    # ---- Save / Test ----
    col_save, col_test = st.columns(2)

    if col_save.button("💾 Save settings", type="primary"):
        if not api_key:
            st.error("Please enter an API key.")
        else:
            try:
                provider = get_provider(
                    name=selected_provider,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                    rate_limit_delay=rate_delay,
                )
                st.session_state.provider = provider
                st.session_state.provider_name = selected_provider
                st.session_state.api_key = api_key
                st.session_state.model_name = model
                # Clear chat engine so it picks up new provider
                st.session_state.chat_engine = None
                st.success(f"✅ Settings saved. Using {PROVIDER_LABELS[selected_provider]} / {model}.")
            except Exception as e:
                st.error(f"Could not initialise provider: {e}")

    if col_test.button("🔌 Test connection"):
        if not api_key:
            st.error("Please enter an API key first.")
        else:
            with st.spinner("Testing connection…"):
                try:
                    provider = get_provider(
                        name=selected_provider,
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=int(max_tokens),
                    )
                    if provider.validate_key():
                        st.success(f"✅ Connection to {PROVIDER_LABELS[selected_provider]} successful!")
                    else:
                        st.error("❌ Connection failed. Check your API key and try again.")
                except ImportError as e:
                    st.error(f"Missing package: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ---- Reload prompts ----
    st.subheader("Prompts")
    st.markdown(
        "Prompts are stored in `config/prompts.yaml`. Edit that file to customise "
        "how the AI analyses your surveys, then click the button below to reload."
    )
    if st.button("🔄 Reload prompts from config/prompts.yaml"):
        load_prompts.clear()
        st.session_state.prompts = load_prompts()
        st.success("✅ Prompts reloaded.")

    # ---- Clear session ----
    st.divider()
    st.subheader("Reset")
    if st.button("🗑️ Clear all session data", type="secondary"):
        keys_to_clear = [
            "survey_data", "current_survey_name", "analysis_results",
            "provider", "api_key", "provider_name", "model_name",
            "chat_history", "chat_engine",
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared. Reload the page to start fresh.")


# ===========================================================================
# Main entrypoint
# ===========================================================================

def main() -> None:
    init_session_state()
    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "📤 Upload Surveys":
        page_upload()
    elif page == "📊 Analyse Survey":
        page_analyse()
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()
