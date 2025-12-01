#!/bin/bash
# Fast dependency installer - creates venv and installs there

set -e

cd /home/kush-mahajan/elodin/examples/rocket-barrowman

echo "📦 Setting up Python environment..."
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check if uv is available (faster than pip)
if command -v uv &> /dev/null; then
    echo "✅ Using uv (faster installer)..."
    uv pip install -r requirements.txt
    echo "✅ Done!"
elif command -v pip3 &> /dev/null; then
    echo "✅ Using pip3..."
    pip3 install --no-cache-dir -r requirements.txt
    echo "✅ Done!"
elif command -v pip &> /dev/null; then
    echo "✅ Using pip..."
    pip install --no-cache-dir -r requirements.txt
    echo "✅ Done!"
else
    echo "❌ No pip or uv found!"
    exit 1
fi

echo ""
echo "✅ All dependencies installed in venv!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Or use: ./run_streamlit.sh"
