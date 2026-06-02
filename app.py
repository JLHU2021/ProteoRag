#!/usr/bin/env python3
"""Entry point for running the Streamlit app.

Usage:
    python app.py
    # or
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from proteomics_rag.ui.app import *  # noqa: F401,F403
