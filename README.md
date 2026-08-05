# WSRL: Warm-Start Reinforcement Learning

<p align="center">
  <a href="http://arxiv.org/abs/2412.07762">
    <img src="https://img.shields.io/badge/arXiv-2412.07762-df2a2a.svg?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
  </a>
  <a href="https://zhouzypaul.github.io/wsrl">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page">
  </a>
</p>

This repository contains the implementation for the paper:

> **Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data**  
> Zhiyuan Zhou, Andy Peng, Qiyang Li, Sergey Levine, and Aviral Kumar  
> **arXiv 2024**  
> https://arxiv.org/abs/2412.07762

The repository provides implementations of **WSRL (Warm-Start Reinforcement Learning)** together with several popular actor-critic reinforcement learning algorithms implemented in **JAX** and **Flax**, including:

- WSRL
- IQL
- CQL
- SAC

---

# Extension

This project has been extended to support **Minari** offline datasets and additional MuJoCo environments.

The following environments are supported:

- Hopper
- Walker2d
- HalfCheetah
- Humanoid

---

## Supported Offline Dataset

The extension replaces the original D4RL datasets with **Minari** datasets.

For more information:

https://minari.farama.org/

---

# Overview

<p align="center">
  <img src="https://zhouzypaul.github.io/images/paper-images/wsrl/teaser.png" alt="Overview" width="900"/>
</p>

---

# Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Remove incompatible JAX packages if already installed:

```bash
pip uninstall -y jax jaxlib nvidia-cudnn-cu12 nvidia-cublas-cu12
```

Install the CUDA 12 compatible JAX version:

```bash
pip install "jax[cuda12_pip]==0.4.23" "nvidia-cudnn-cu12<9.0" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

# Running

The main training script is:

```text
finetune.py
```

Configuration files are located at:

Shared algorithm configurations:

```text
experiments/configs/
```

Environment-specific training configurations:

```text
experiments/configs/train_config.py
```

## Running Commands

Complete commands for reproducing all experiments on the supported Minari environments are available in **[DLL.md](DLL.md)**.

Supported environments:

- Hopper
- Walker2d
- HalfCheetah
- Humanoid

---

# Contribution Breakdown

| Task | Contributor(s) |
|------|----------------|
| Reproduction of Hopper experiments | Kartik |
| Reproduction of Walker2D experiments | Nitishkumar |
| Reproduction of HalfCheetah experiments | Niladri |
| Humanoid extension (IQL experiments) | Nitishkumar |
| Humanoid extension (CQL experiments) | Niladri |
| Humanoid extension (WSRL experiments) | Kartik |
| Ablation 3.1 (Warmup Steps) | Niladri |
| Ablation 3.2 (Warmup Buffer Initialization) | Kartik |
| Ablation 3.3 (Value Initialization) | Kartik |
| Ablation 3.4 (Offline Data Retention) | Kartik |
| Ablation 3.5 (Reward Configuration) | Nitishkumar |
| Repository organization and code cleanup | Kartik |
| Presentation slides | Kartik, Niladri, Nitishkumar |
| Report writing | Nitishkumar, Kartik |
| Poster preparation | Niladri, Kartik |

---

# Citation

If you use this work, please cite:

```bibtex
@article{zhou2024efficient,
  author       = {Zhiyuan Zhou and Andy Peng and Qiyang Li and Sergey Levine and Aviral Kumar},
  title        = {Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data},
  conference   = {arXiv Pre-print},
  year         = {2024},
  url          = {http://arxiv.org/abs/2412.07762},
}
```

---

# Credits

The original repository was developed by:

- Zhiyuan Zhou
- Andy Peng
- Qiyang Li
- Sergey Levine
- Aviral Kumar

The implementation is built upon a version of **jaxrl_minimal** by Dibya Ghosh, with contributions from:

- Dibya Ghosh
- Kevin Black
- Homer Walke
- Kyle Stachowicz
- and other contributors.

This repository extends the original implementation by adding **Minari dataset support** for offline reinforcement learning and support for the following MuJoCo environments:

- Hopper
- Walker2d
- HalfCheetah
- Humanoid