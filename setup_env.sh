#!/bin/bash

# Nazwa wirtualnego środowiska
ENV_DIR="albutortion-env"


if [ ! -f "requirements.txt" ]; then
    echo "File requirements.txt does not exist. Create this file and add required packages (e.g. opencv-python, albumentations, matplotlib)."
    exit 1
fi


if [ -d "$ENV_DIR" ]; then
    echo "Virtual environment already exists. Activating the environment..."
    source "$ENV_DIR/bin/activate"
else
    eco "Create virtual environment..."
    python3 -m venv "$ENV_DIR"

    echo "Activating the virtual environment..."
    source "$ENV_DIR/bin/activate"

    echo "Upgrade pip..."
    pip install --upgrade pip
    source "$ENV_DIR/bin/activate"
fi
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "Enviroment is ready."