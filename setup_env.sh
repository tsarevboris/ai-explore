#!/usr/bin/env bash
set -e

PYTHON=${1:-python}

if [ ! -d ".venv" ]; then
  echo "Creating venv..."
  $PYTHON -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

KERNEL_NAME=$(basename "$PWD")

echo "Registering Jupyter kernel: $KERNEL_NAME"
python -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "Python ($KERNEL_NAME)"

echo "Done."