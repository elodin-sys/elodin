#!/bin/bash
# Run Streamlit app - activates venv automatically

set -e

cd /home/kush-mahajan/elodin/examples/rocket-barrowman

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./install_deps.sh first"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if packages are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found in venv"
    echo "   Installing dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install -r requirements.txt
    else
        pip3 install --no-cache-dir -r requirements.txt
    fi
fi

echo "🚀 Starting Streamlit..."
echo ""

streamlit run app.py

