# GPRO Empathy: Group Relative Policy Optimization for Empathy Training

A structured project for training empathetic language models using Group Relative Policy Optimization (GRPO) with specialized reward functions for empathy and semantic similarity.

## Overview

This project implements GRPO training for developing more empathetic conversational AI models. It uses:

- **Semantic Similarity Reward**: Measures how well the model response aligns with user input
- **Empathy Model Reward**: Uses a fine-tuned RoBERTa model to evaluate empathy levels
- **WASSA Empathy Dataset**: Training data from conversation turns labeled for empathy

## Features

- 🤖 **GRPO Training**: Efficient training using Group Relative Policy Optimization
- 💝 **Empathy Rewards**: Specialized reward functions for empathetic responses  
- 🎯 **LoRA Fine-tuning**: Memory-efficient training with Low-Rank Adaptation
- 🚀 **vLLM Inference**: Fast inference with vLLM backend
- ⚙️ **Configurable**: YAML-based configuration system
- 📊 **Interactive Inference**: Chat interface for testing trained models

## Project Structure

```
GPRO-EMPATHY/
├── src/gpro_empathy/           # Main package
│   ├── data/                   # Dataset loading utilities
│   ├── models/                 # Reward functions
│   ├── training/               # GRPO trainer
│   └── utils/                  # Inference utilities
├── scripts/                    # Training and inference scripts
├── configs/                    # Configuration files
├── notebooks/                  # Original development notebooks
└── requirements.txt            # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GPRO-EMPATHY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training

Train the model with default configuration:

```bash
python scripts/train.py
```

Or with custom config:

```bash
python scripts/train.py --config configs/training_config.yaml --max-steps 500
```

### Inference

Single message response:

```bash
python scripts/inference.py --message "I'm feeling really overwhelmed with work lately."
```

Interactive chat mode:

```bash
python scripts/inference.py --interactive
```

## Configuration

Edit `configs/training_config.yaml` to customize:

- Model parameters (LoRA rank, sequence length)
- Training hyperparameters (learning rate, batch size)
- Reward function settings
- Inference parameters

## Dataset

Uses the `miladsolo/wassa-conv-turn-empathy` dataset, which contains:
- Conversation turns with empathy labels (0-5 scale)  
- System prompt for XML-formatted empathetic responses
- Automatic data preprocessing and filtering

## Reward Functions

### Semantic Similarity Reward
- Uses `cross-encoder/stsb-roberta-large`
- Measures alignment between user input and model response
- Calibrated to [0,1] range

### Empathy Model Reward  
- Uses `miladsolo/roberta-lora-wassa-empathy`
- Predicts empathy level in model responses
- Based on WASSA empathy classification

## Model Architecture

- **Base Model**: meta-llama/meta-Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA with rank 32
- **Training**: GRPO with multiple reward functions
- **Inference**: vLLM for fast generation

## Output Format

The model generates responses in XML format:

```xml
<reasoning>
- User is expressing work stress and overwhelm
- Emotion: anxiety/stress, intensity: 3-4  
- Plan: validate feelings and offer gentle support
</reasoning>
<answer>
I hear how overwhelming work has been feeling for you lately - that kind of stress can be really draining. Have you been able to take any small breaks for yourself during the day?
</answer>
```

## Advanced Usage

### Custom Reward Functions

Add your own reward functions in `src/gpro_empathy/models/reward_functions.py`:

```python
def my_custom_reward(prompts, completions, **kwargs) -> list[float]:
    # Your reward logic here
    return [score for completion in completions]
```

### Batch Inference

```python
from gpro_empathy.utils.inference import EmpathyInference

inference = EmpathyInference(
    config_path="configs/training_config.yaml",
    lora_path="grpo_saved_lora"
)

messages = ["I'm sad", "I'm excited", "I'm confused"]
responses = inference.batch_generate(messages)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for training
- unsloth, vLLM, transformers, TRL

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{gpro-empathy,
  title={GPRO Empathy: Group Relative Policy Optimization for Empathy Training},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gpro-empathy}
}
```