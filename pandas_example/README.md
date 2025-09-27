Streamlit UI for the one-cycle account deduper in `example_run.py`.

Usage
-----
1. Create a virtual environment and install dependencies:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2. Run the app:

   streamlit run streamlit_app.py

Notes
-----
- If Azure OpenAI env vars (AOAI_ENDPOINT, AOAI_CHAT_DEPLOYMENT, AOAI_EMBEDDING_DEPLOYMENT) are not set, the app uses a deterministic mock LLM and embedding generator.
- The app stores cumulative mapping and reviewer notes in `output/azure_cicd_cli/` as defined by `example_run.py`.
- This is a single-cycle UI for interactive testing. It does not yet implement multi-run orchestration beyond saving mappings and notes.
