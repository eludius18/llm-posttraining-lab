# SFT (Supervised Fine-Tuning) Pipeline

A complete pipeline for Supervised Fine-Tuning with QLoRA

## Overview

This module provides a comprehensive implementation of Supervised Fine-Tuning (SFT) for Large Language Models using QLoRA (Quantized Low-Rank Adaptation). It's designed to be efficient, configurable, and easy to use for fine-tuning models on custom datasets.

### Key Features

- **QLoRA Integration**: Efficient fine-tuning with 4-bit quantization
- **Multiple Model Support**: GPT-2, LLaMA, Mistral, Qwen architectures
- **Configurable Training**: Environment-based configuration
- **Automatic Setup**: One-command installation and setup
- **Production Ready**: Clean code structure with proper error handling

## Project Structure

```
src/sft/
├── sft_pipeline.py          # Main SFT pipeline implementation
├── __main__.py              # Module entry point
├── data/                    # Dataset directory
│   └── simple_dataset.jsonl # Example dataset
├── helpers/                 # Setup and configuration helpers
│   ├── setup.py             # Automatic environment setup
│   ├── create_env.py        # Environment file creator
│   └── env_example.txt      # Configuration template
├── sft_model/               # Training checkpoints (generated)
├── sft_model_final/         # Final model (generated)
└── README.md                # This file
```

## Installation

```bash
cd src/sft
python helpers/setup.py
```

This will automatically:
- Create a virtual environment (`sft_env/`)
- Install all dependencies from `requirements.txt`
- Provide next steps

## Configuration

### 1. Create Environment File

```bash
python helpers/create_env.py
```

This creates a `.env` file from the template. Edit it to customize your configuration:

```bash
# Model Configuration
MODEL_NAME=gpt2
USE_GPU=false

# Dataset Configuration
DATASET_PATH=data/simple_dataset.jsonl

# Training Configuration
OUTPUT_DIR=./sft_model
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-4
WARMUP_RATIO=0.05
LOGGING_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500

# LoRA Configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Generation Configuration
MAX_NEW_TOKENS=100
DO_SAMPLE=false
```

### 2. Prepare Your Dataset

Your dataset should be in JSONL format with the following structure:

```json
{"instruction": "Create a function to calculate the sum of a sequence of integers", "input": "[1, 2, 3, 4, 5]", "output": "# Python code\ndef sum_sequence(sequence):\n    sum = 0\n    for num in sequence:\n        sum += num\n    return sum"}
{"instruction": "Explain what machine learning is", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."}
```

## Usage

```bash
source sft_env/bin/activate
python __main__.py
```

## Configuration Options

### Model Configuration
- `MODEL_NAME`: Hugging Face model name (default: gpt2)
- `USE_GPU`: Enable GPU training (default: false)

### Training Configuration
- `NUM_TRAIN_EPOCHS`: Number of training epochs (default: 1)
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size per device (default: 1)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps (default: 16)
- `LEARNING_RATE`: Learning rate (default: 2e-4)
- `WARMUP_RATIO`: Warmup ratio (default: 0.05)

### LoRA Configuration
- `LORA_R`: LoRA rank (default: 16)
- `LORA_ALPHA`: LoRA alpha (default: 32)
- `LORA_DROPOUT`: LoRA dropout (default: 0.05)

## Output

The pipeline generates:

- `sft_model/`: Training checkpoints with optimizer states
- `sft_model_final/`: Final fine-tuned LoRA adapter for inference
- Training logs and metrics

### Model Files

The final model contains:
- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: LoRA configuration
- Tokenizer files: `vocab.json`, `tokenizer.json`, etc.