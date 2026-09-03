#!/usr/bin/env bash
set -e

echo "========================================================================="
echo "  PhytoCuckoo - Setting up Python Virtual Environment for Feature Extractor"
echo "========================================================================="
echo ""

PYTHON_BIN=""
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3 was not found. Please install Python 3."
    exit 1
fi

echo "[OK] Using Python: $($PYTHON_BIN --version)"
echo "Creating virtual environment in backend/extractor/venv ..."
$PYTHON_BIN -m venv backend/extractor/venv

echo "Installing PyTorch CPU and Extractor dependencies..."
backend/extractor/venv/bin/pip install --upgrade pip
backend/extractor/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
backend/extractor/venv/bin/pip install -r backend/extractor/requirements.txt

echo ""
echo "========================================================================="
echo "  Feature Extractor environment is ready!"
echo "  The backend will automatically detect and use backend/extractor/venv."
echo "========================================================================="
