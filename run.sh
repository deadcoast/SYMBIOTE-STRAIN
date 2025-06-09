#!/bin/bash
# This script activates the virtual environment and runs the Pygame viewer.

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the directory where the script is located.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Activate the virtual environment.
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the python script.
python "$SCRIPT_DIR/scripts/run_pygame.py" 