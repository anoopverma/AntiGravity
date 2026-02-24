#!/usr/bin/env bash

# Navigate to the correct directory
cd "$(dirname "$0")"

# Activate the virtual environment if it exists, otherwise just try to run it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting AntiGravity Dashboard Engine..."
python app.py
