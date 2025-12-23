#!/bin/bash
# Run Streamlit app
# Prerequisites: Run from repo root with .venv activated (see README.md quickstart)

set -e

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Streamlit..."
echo "   Open http://localhost:8501 in your browser"
echo ""

streamlit run app.py
