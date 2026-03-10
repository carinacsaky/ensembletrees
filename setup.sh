#!/usr/bin/env bash
# Create virtual environment and install dependencies.
# Run once: bash setup.sh
# Activate afterwards: source .venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Creating virtual environment in .venv ..."
python3 -m venv .venv

echo "==> Upgrading pip ..."
.venv/bin/pip install --upgrade pip --quiet

echo "==> Installing dependencies from requirements.txt ..."
.venv/bin/pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run the script with:"
echo "  python ensembletrees.py"
