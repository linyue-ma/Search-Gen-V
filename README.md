# Search-Gen-V

This project introduces the **“nugget-as-rubric”** paradigm, which decomposes information into atomic, independently verifiable units to build a unified reward framework for **verifiable evaluation**.

- 🧩 **Short-form tasks**: each *nugget* corresponds to a single evaluation criterion.  
- 📜 **Long-form tasks**: multiple *nuggets* collectively form a multi-dimensional rubric.

Based on this paradigm, we develop **Search-Gen-V**, a lightweight **generative verifier** that:

- 🔍 Automatically constructs evaluation rubrics and efficiently determines the alignment between text and criteria;  
- 🧠 Produces structured reasoning outputs to enhance transparency and interpretability;  
- ⚡ Supports batch, multi-rubric verification, achieving high accuracy with reduced computational cost.

The proposed framework demonstrates **strong interpretability, robustness, and scalability** across both short- and long-form workloads, providing a **unified, reliable, and efficient reward modeling solution** for **Search-Augmented LLMs** in **Verifiable Reinforcement Learning (RLVR)**.

---

## Table of Contents
- [项目简介](#项目简介)
- [环境依赖](#环境依赖)
- [安装与运行](#安装与运行)
- [数据说明](#数据说明)
- [模型训练](#模型训练)
- [评估与结果](#评估与结果)
- [引用](#引用)
- [联系方式](#联系方式)

---

## 项目简介

**Search-Gen-V** is an open-source framework for **verifiable reward modeling** in Search-Augmented LLMs.  
It introduces the *nugget-as-rubric* paradigm, which transforms fine-grained information nuggets into structured evaluation rubrics for both short- and long-form reasoning tasks.  
Built upon **Qwen3-4B** and the **veRL** framework, Search-Gen-V unifies **SFT, RM, and PPO** stages under a verifiable learning pipeline, enabling models to reason, verify, and self-improve through structured feedback.
---

## ⚙️ 环境依赖
```bash
# 基础环境
python>=3.10
pytorch>=2.2
transformers>=4.40
accelerate
verl

# 可选工具
wandb
faiss
uvicorn
>>>>>>> a02b072f (readme commit)
