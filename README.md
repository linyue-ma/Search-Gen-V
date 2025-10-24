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
- data-tain: https://huggingface.co/datasets/lnm1p/Search-Gen-V
- data-raw: https://huggingface.co/datasets/lnm1p/Search-Gen-V-raw
- data-eval: https://huggingface.co/datasets/lnm1p/Search-Gen-V-eval
- model:
  - Search-Gen-V-4B:https://huggingface.co/lnm1p/search-gen-v-4b
  - Search-Gen-V-1.7B-SFT:https://huggingface.co/lnm1p/search-gen-v-1.7b-sft

---

## nugget-as-rubric

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
### 1. Data
You can generate personalized system messages and user messages for the data-raw dataset available at [Search-Gen-V-raw](https://huggingface.co/datasets/lnm1p/Search-Gen-V-raw) using the following procedure:
- For SFT:
```bash
python Search-Gen-V/data/preprocess/sft_data.py
```
- For DAPO:
```bash
python Search-Gen-V/data/preprocess/dapo_data.py
```
You can also directly train your model using the data-train dataset: [Search-Gen-V](https://huggingface.co/datasets/lnm1p/Search-Gen-V).
### 2. SFT
```bash
bash Search-Gen-V/trainer/train_sft.sh
```

### 3. DAPO
```bash
bash Search-Gen-V/trainer/train_dapo.sh
```
---

## Evaluator

The evaluator module is used to assess model outputs based on the nugget-as-rubric paradigm.  
Detailed instructions and usage examples can be found at: `/Search-Gen-V/evaluator/eval/README.md`.<br>
The evaluation input data is provided by data-eval: [Search-Gen-V-eval](https://huggingface.co/datasets/lnm1p/Search-Gen-V-eval).
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
## Result
**Table 1. Results on the eval set**
| **Verifier Model** | **Rubric Precision** | **Rubric Recall** | **Rubric F1** | **Sample Precision** | **Sample Recall** | **Sample F1** | **Avg. F1** |
|---------------------|---------------------|------------------|---------------|----------------------|-------------------|---------------|-------------|
| Qwen3-1.7B | 0.41 | 0.49 | 0.34 | 0.48 | 0.40 | 0.32 | 0.33 |
| Qwen2.5-3B | 0.42 | 0.47 | 0.43 | 0.49 | 0.46 | 0.43 | 0.43 |
| Qwen3-4B | 0.56 | 0.62 | 0.57 | 0.61 | 0.58 | 0.58 | 0.58 |
| Qwen3-8B | 0.54 | 0.66 | 0.55 | 0.62 | 0.61 | 0.57 | 0.56 |
| LLaMA-3.1-8B | 0.45 | 0.54 | 0.42 | 0.34 | 0.41 | 0.32 | 0.37 |
| Qwen3-30B-A3B | 0.56 | 0.66 | 0.56 | 0.63 | 0.62 | 0.62 | 0.58 |
| Qwen2.5-32B-Instruct | 0.60 | 0.67 | 0.60 | 0.67 | 0.68 | 0.64 | 0.62 |
| **Search-Gen-V-1.7B (SFT)** | **0.63** | **0.62** | **0.62** | **0.66** | **0.66** | **0.66** | **0.64** |
| **Search-Gen-V-4B (SFT)** | **0.70** | **0.66** | **0.68** | **0.72** | **0.72** | **0.71** | **0.70** |
| **Search-Gen-V-4B (SFT+RL)** | **0.71** | **0.68** | **0.70** | **0.74** | **0.74** | **0.73** | **0.72** |
| Qwen3-235B-A22B-Instruct-2507 | 0.72 | 0.73 | 0.73 | 0.76 | 0.76 | 0.76 | 0.74 |

**Table 2. Accuracy comparison on verifying rubrics in longform answers from DeepResearch Bench**
| **Verifier Model**      | **Precision** | **Recall** | **F1** |
|-------------------------|---------------|------------|--------|
| Qwen3-4B                | 0.42          | 0.56       | 0.42   |
| **Search-Gen-V-4B**     | **0.59**      | 0.57       | 0.57   |
| Qwen3-235B-A22B         | 0.57          | **0.67**   | **0.61** |

**Table 3. Results on the short-form workload, HotpotQA**
| **Verifier Model**          | **Precision** | **Recall** | **F1** |
|-----------------------------|---------------|------------|--------|
| EM                          | 0.84          | **0.80**   | **0.82** |
| Qwen3-4B                    | 0.83          | 0.70       | 0.71    |
| **Search-Gen-V-4B**         | 0.86          | 0.76       | 0.77    |
| Qwen3-235B-A22B             | **0.87**      | 0.78       | 0.80    |
| EM + Qwen3-4B               | 0.94          | 0.92       | 0.93    |
| **EM + Search-Gen-V-4B**    | 0.95          | 0.93       | 0.94    |
| EM + Qwen3-235B-A22B        | **0.96**      | **0.94**   | **0.95** |
---

## Citation

```bibtex
@article{ma2025searchgenv,
  title={AN EFFICIENT RUBRIC-BASED GENERATIVE VERIFIER FOR SEARCH-AUGMENTED LLMS},
  author={Ma, Linyue and Xu, Yilong and Long, Xiang and Zheng, Zhi},
  journal={arXiv preprint arXiv:2510.14660},
  year={2025},
  url={https://arxiv.org/abs/2510.14660}
}
```
