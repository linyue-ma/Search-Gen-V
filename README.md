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

