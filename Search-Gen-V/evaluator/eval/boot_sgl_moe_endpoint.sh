#!/bin/bash
# Launch router with 4 workers

MODEL_PATH=$1
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp-size 8 \
    --ep-size 8 \
    --max-running-requests 160 \
    --cuda-graph-max-bs 160 \
    --mem-fraction-static 0.8 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
