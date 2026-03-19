# LLM Training & Inference Resource Calculator

A Python-based utility to estimate VRAM requirements for Large Language Models. This tool accounts for model architecture, optimizer states (Adam), DeepSpeed ZeRO-3 sharding, and LoRA adaptation to help you plan your hardware allocation.

## 🚀 Quick Start with `uv`

This project is optimized for [uv](https://github.com/astral-sh/uv). You don't need to manually create virtual environments or install dependencies; `uv` handles it on the fly.

### 1. Setup for your config
Edit the `config.yaml` to match the model type and setup you're using.

### 2. Run the Calculator
Ensure `config.yaml` is in the root directory, then run:
```bash
uv run calculator.py
