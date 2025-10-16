# Search-Gen-V

This project introduces the **“nugget-as-rubric”** paradigm, which decomposes information into atomic, independently verifiable units to build a unified reward framework for **verifiable evaluation**.

- **Short-form tasks**: each *nugget* corresponds to a single evaluation criterion.  
- **Long-form tasks**: multiple *nuggets* collectively form a multi-dimensional rubric.

Based on this paradigm, we develop **Search-Gen-V**, a lightweight **generative verifier** that:

- Automatically constructs evaluation rubrics and efficiently determines the alignment between text and criteria;  
- Produces structured reasoning outputs to enhance transparency and interpretability;  
- Supports batch, multi-rubric verification, achieving high accuracy with reduced computational cost.

The proposed framework demonstrates **strong interpretability, robustness, and scalability** across both short- and long-form workloads, providing a **unified, reliable, and efficient reward modeling solution** for **Search-Augmented LLMs** in **Verifiable Reinforcement Learning (RLVR)**.

---

## Datasets

We have already uploaded the data and models to Hugging Face. You can access them and modify the corresponding path variables in the running script:
- data: https://huggingface.co/datasets/lnm1p/Search-Gen-V
- model: https://huggingface.co/lnm1p/search-gen-v-4b

---

## Installation

### 1. Update Python and Configure a Virtual Environment

```bash
# Update package lists
sudo apt update

# Install Python 3.10 and venv module
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create a virtual environment
python3.10 -m venv ~/.python/search-gen-v

# Activate the virtual environment
source ~/.python/search-gen-v/bin/activate
```

### 2. Install veRL and Dependencies

```bash
# Clone the repository
cd ~
git clone https://github.com/linyue-ma/Search-Gen-V.git
cd Search-Gen-V

# Install veRL
pip install .

# Install additional requirements
pip install -r ./requirements_sglang.txt

# Manually install flash-attn dependencies
pip install wheel packaging
pip install flash-attn --no-build-isolation --no-deps
```
---

## Quick start

### 1. SFT
```bash
bash Search-Gen-V/trainer/train_sft.sh
```

### 2. DAPO
```bash
bash Search-Gen-V/trainer/train_dapo.sh
```
---

## Evaluator

The evaluator module is used to assess model outputs based on the nugget-as-rubric paradigm.  
Detailed instructions and usage examples can be found at: `/Search-Gen-V/evaluator/eval/README.md`
###  Quick Start

#### 1. Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup the environment
./setup.sh
source .venv/bin/activate
```

#### 2. Generate Configuration
```bash
# Generate configuration templates
nugget-eval --generate-config config/my_thinking.yaml --template-type thinking
nugget-eval --generate-config config/my_multi.yaml --template-type multi_run
```

#### 3. Configure Your Evaluation
Edit your configuration file:
```yaml
model:
  base_url: "http://localhost:8000/v1"
  name: "/path/to/your/model"
  format_type: "adaptive"        # Controls prompt format
  error_handling: "sequential"   # Preserve partial results
  enable_thinking: true

data:
  input_path: "/path/to/input.jsonl"
  gold_path: "/path/to/gold.jsonl"

evaluation:
  num_runs: 1          # Single run (1) or multi-run (16+)
  batch_size: 10
  num_workers: 8
```

#### 4. Run Evaluation
```bash
# Single-run evaluation
nugget-eval --config config/my_thinking.yaml

# Multi-run statistical analysis  
nugget-eval --config config/my_multi.yaml --num-runs 16
```
---

## Citation

@article{ma2025searchgenv,
  title={AN EFFICIENT RUBRIC-BASED GENERATIVE VERIFIER FOR SEARCH-AUGMENTED LLMS},
  author={Ma, Linyue and Xu, Yilong and Long, Xiang and Zheng, Zhi},
  journal={Proceedings of XYZ Conference},
  year={2025},
  url={https://github.com/linyue-ma/Search-Gen-V}
}
