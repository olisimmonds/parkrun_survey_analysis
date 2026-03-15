# Parkrun Survey Analysis

An open-source tool that uses AI to extract insights from parkrun surveys.
Built by a Digital Ambassador volunteer to help parkrun understand participant and
volunteer feedback at scale.

---

## What this does

- Parses **SurveyMonkey** CSV and XLSX exports automatically
- Extracts **keywords** and **statistics** from every survey question
- Uses an **LLM** to identify themes, sentiment, and actionable recommendations
- Provides a **chat interface** to ask natural-language questions about your survey data

---

## Who it's for

This tool is designed to be picked up and run by anyone, including volunteers with
no coding background. All configuration is in simple text files, and the interface
is a web app that runs in your browser.

---

## Quick start (5 minutes)

### 1. Install Python

If you don't have Python, download it from [python.org/downloads](https://www.python.org/downloads/).
Choose Python 3.10 or higher. During installation, tick **"Add Python to PATH"**.

### 2. Clone or download this repository

**Option A — with Git:**
```bash
git clone https://github.com/your-username/parkrun_survey_analysis.git
cd parkrun_survey_analysis
```

### 3. Install dependencies in virtual enviroment

```bash
pip install -r requirements.txt
```

This installs Streamlit, pandas, and the LLM provider packages. It takes 1–3 minutes.

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

A browser tab will open automatically at `http://localhost:8501`.

### 5. Add your API key

Go to **Settings** in the sidebar and add an API key for your chosen LLM provider.
See [docs/llm_setup.md](docs/llm_setup.md) for step-by-step instructions — Groq is free
and is the recommended starting point.

---

## Connecting an LLM

Three providers are supported. All require an API key entered through the Settings page.
**Keys are never saved to disk.**

| Provider | Cost | Notes |
|----------|------|-------|
| **Groq** | Free | Best for getting started. Uses Llama 3 models. |
| **OpenAI** | ~£0.001–0.3 per analysis | gpt-4o-mini recommended |
| **Anthropic** | ~£0.001–0.3 per analysis | claude-haiku recommended |

Full setup instructions: [docs/llm_setup.md](docs/llm_setup.md)

---

## Using the tool

### Upload a survey

1. Export your survey from SurveyMonkey:
   - Download as CSV or XLSX
2. Go to **Upload Surveys** in the sidebar
3. Upload the file — it's saved to `data/surveys/` in this repository

### Run analysis

1. Go to **Analyse Survey**
2. Select your survey from the dropdown
3. Click **Run Basic Stats** for statistics and keywords (no LLM needed)
4. Click **Run Full Analysis** for AI-powered insights (requires API key)

### Chat with your data

After running analysis, a chat interface appears at the bottom of the Analyse page.
Ask questions like:
- *"What themes appear most frequently?"*
- *"What are participants most positive about?"*
- *"Are there any concerns about accessibility?"*
- *"What do volunteers say about their experience?"*

---

## Repository structure

```
parkrun_survey_analysis/
│
├── app/
│   └── streamlit_app.py      # Main web interface
│
├── src/
│   ├── survey_parser.py      # Parses SurveyMonkey CSV/XLSX exports
│   ├── analysis_engine.py    # Coordinates stats, keywords, and LLM calls
│   ├── llm_interface.py      # Abstraction layer for LLM providers
│   └── keyword_extraction.py # TF-IDF keyword extraction
│
├── config/
│   ├── settings.yaml         # App and LLM configuration
│   └── prompts.yaml          # Parkrun-specific LLM prompts (edit these!)
│
├── data/
│   └── surveys/              # Uploaded survey files (git-ignored)
│
├── docs/
│   └── llm_setup.md          # Step-by-step LLM connection guide
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Customising prompts

The AI prompts are stored in `config/prompts.yaml`. They include a parkrun-specific
system prompt and question-type-specific templates for free-text, rating, and choice questions.

To customise:
1. Open `config/prompts.yaml` in any text editor
2. Edit the prompt text — `{question}`, `{responses}`, `{stats}`, and `{keywords}` are
   automatically replaced with the actual data
3. In the app, go to **Settings** and click **Reload prompts** to apply your changes

---

## Configuration

All settings are in `config/settings.yaml`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.default_provider` | `groq` | Which LLM to use by default |
| `llm.providers.groq.default_model` | `llama-3.3-70b-versatile` | Groq model name |
| `llm.providers.openai.default_model` | `gpt-4o-mini` | OpenAI model name |
| `analysis.max_verbatim_samples` | `5` | Max sample responses sent to LLM |
| `analysis.keywords_count` | `15` | Keywords extracted per question |
| `parser.custom_stopwords` | See file | Words excluded from keywords |

---

## About this project

This tool was built by a [parkrun Digital Ambassador](https://www.parkrun.com) volunteer
to explore how AI and data science can help parkrun generate better insights from their
large volume of community surveys.

parkrun is a free, community event where you can walk, jog, run, volunteer or spectate.
parkrun is 5k and takes place every Saturday morning. junior parkrun is 2k, dedicated to
4-14 year olds and their families, every Sunday morning. parkrun is positive, welcoming
and inclusive, there is no time limit and no one finishes last. Everyone is welcome to
come along.

---

## Contributing

Contributions are welcome. To report a bug or suggest a feature, open an issue on GitHub.

---

## Privacy and data

- Uploaded survey files are stored **only on your local machine** in `data/surveys/`
- Survey data is **never stored in the cloud** by this application
- When using LLM analysis, a **small anonymised sample** of responses is sent to your
  chosen LLM provider's API — check their privacy policy for details
- API keys are held **in your browser session only** and are never written to any file

---

## To Do

- Explore using vector embeding to analysis grater number of surveys