#!/bin/bash
# Convenient script to run evaluations with proper environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE} Nugget Evaluation Runner${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED} Virtual environment not found. Run setup.sh first.${NC}"
    exit 1
fi

# Check if we're in the virtual environment
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo -e "${YELLOW}⚡ Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Default config
CONFIG_FILE="config/thinking_mode.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config CONFIG_FILE   Configuration file to use (default: config/thinking_mode.yaml)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default thinking mode config"
            echo "  $0 -c config/multi_run.yaml          # Use multi-run config"
            echo "  $0 -c config/my_custom.yaml          # Use custom config"
            exit 0
            ;;
        *)
            echo -e "${RED} Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED} Configuration file not found: $CONFIG_FILE${NC}"
    echo ""
    echo -e "${YELLOW} Generate a config file first:${NC}"
    echo "   nugget-eval --generate-config $CONFIG_FILE"
    exit 1
fi

echo -e "${GREEN} Using configuration: $CONFIG_FILE${NC}"
echo ""

# Run the evaluation
echo -e "${BLUE} Starting evaluation...${NC}"
nugget-eval --config "$CONFIG_FILE"

echo -e "${GREEN} Evaluation completed!${NC}"