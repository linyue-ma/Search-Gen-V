#!/bin/bash
# Setup script for nugget evaluation environment using uv

set -e  # Exit on error

echo " Setting up nugget evaluation environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo " uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo " uv found"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo " Creating virtual environment..."
    uv venv --python 3.11
    echo " Virtual environment created"
else
    echo " Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo " Installing dependencies..."
uv pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

echo " Installing development dependencies..."
uv pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo " Setup complete!"
echo ""
echo " Next steps:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Generate a config file: nugget-eval --generate-config config/my_config.yaml"
echo "3. Edit the config file with your paths"
echo "4. Run evaluation: nugget-eval --config config/my_config.yaml"
echo ""
echo " Or use the development mode:"
echo "   python -m nugget_eval.cli --generate-config config/test.yaml"