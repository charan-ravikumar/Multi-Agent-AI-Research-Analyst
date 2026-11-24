"""
Streamlit entry point (kept at app/main.py so the Streamlit Cloud "main file path" never changes).

Everything real lives in src/research_analyst/ui/.  This shim only does the things that MUST happen
before the first project import:

  1. put src/ on sys.path (Streamlit Cloud installs requirements.txt, not this project)
  2. set cheaper defaults for the web UI
  3. export Streamlit secrets (incl. ones pasted under a [section]) to the environment, because
     research_analyst.config.Settings reads the environment when first imported
"""
import contextlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("SEARCH_MAX_RESULTS", "5")
os.environ.setdefault("READER_MAX_CONCURRENT_LLM_CALLS", "2")

import streamlit as st  # noqa: E402

from research_analyst.ui.secrets_env import export_secrets_to_env  # noqa: E402

with contextlib.suppress(Exception):   # no secrets configured (local runs use .env)
    export_secrets_to_env(dict(st.secrets), os.environ)

from research_analyst.ui.app import main  # noqa: E402

main()
