# GPRO-EMPATHY: Technical Overview

## System Architecture

### Core Components

1. **Base Model**: Meta-Llama-3.1-8B-Instruct
2. **Adaptation Method**: LoRA (Low-Rank Adaptation) with rank 32
3. **Training Algorithm**: Group Relative Policy Optimization (GRPO)
4. **Reward Functions**: Dual reward system for comprehensive evaluation

### Data Flow

```
Raw WASSA Data → Prompt Formatting → GRPO Training → LoRA Adapter → Inference
      ↓                ↓                   ↓             ↓           ↓
Text utterances → Chat templates → Multi-gen → Reward scores → Empathetic responses
```

## Training Pipeline

### 1. Data Processing
- **Dataset**: `miladsolo/wassa-conv-turn-empathy`
- **Format**: Chat templates with system instructions
- **Structure**: User message → System prompt → Model generation

### 2. Model Configuration
```python
# LoRA Configuration
r=32                    # Low-rank adaptation rank
target_modules=[        # Attention and MLP layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
lora_alpha=32          # LoRA scaling parameter
```

### 3. GRPO Training
- **Generation**: 3 responses per prompt
- **Evaluation**: Dual reward functions
- **Optimization**: Policy gradient updates
- **Efficiency**: Unsloth integration

### 4. Reward System

#### Semantic Similarity Reward
```python
Model: cross-encoder/stsb-roberta-large
Input: (user_utterance, model_answer)
Output: [0,1] similarity score
Calibration: Temperature scaling (T=0.6)
```

#### Empathy Prediction Reward
```python
Model: miladsolo/roberta-lora-wassa-empathy
Input: model_answer_text
Output: [0,1] empathy score
Processing: 3-class logits → empathy dimension
```

## Implementation Details

### XML Response Structure
```xml
<reasoning>
- User's main concern: [analysis]
- Emotional content: [emotions identified]
- Intensity level: [0-5 scale]
- Response approach: [strategy]
</reasoning>
<answer>
[Empathetic response text]
</answer>
```

### Key Design Decisions

1. **Answer Extraction**: Only `<answer>` content evaluated by rewards
2. **Structured Reasoning**: Explicit emotional analysis before response
3. **Graduated Empathy**: Intensity-aware response generation
4. **Efficient Training**: LoRA + 4-bit quantization + Unsloth

### Memory Optimization
- 4-bit quantization for reduced memory usage
- LoRA for parameter-efficient fine-tuning
- Gradient checkpointing for memory efficiency
- vLLM integration for fast inference

## Performance Characteristics

### Training Efficiency
- GPU Memory: ~60% utilization on single GPU
- Training Speed: ~55s per step (3 generations)
- Model Size: 8B parameters with 32-rank LoRA adaptation
- Convergence: 250 training steps

### Response Quality
- Contextual relevance through semantic similarity
- Emotional appropriateness through empathy modeling
- Structured reasoning for interpretability
- Graduated responses based on emotional intensity

## Integration Points

### Modular Design
- Pluggable reward functions
- Configurable training parameters
- Extensible prompt templates
- Flexible inference backends

### API Structure
```python
# Training
trainer = GPROEmpathyTrainer(model_name, config)
trainer.setup_training(reward_funcs=[semantic_reward, empathy_reward])
trainer.train()

# Inference
inference = EmpathyInference(config_path, lora_path)
response = inference.generate_empathetic_response(user_message)
```

## Monitoring and Debugging

### Training Metrics
- Reward scores per generation
- Loss convergence tracking
- Generation quality assessment
- Memory usage monitoring

### Debug Capabilities
- Completion pipeline tracking
- Reward function introspection
- XML parsing validation
- Answer extraction verification

---

This technical overview provides implementation details for developers working with the GPRO-EMPATHY system.