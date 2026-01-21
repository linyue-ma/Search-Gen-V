#!/bin/bash

MODEL_PATH="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp-size 8 \
    --ep-size 8 \
    --max-running-requests 160 \
    --cuda-graph-max-bs 160 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
    # --mem-fraction-static 0.8 \
