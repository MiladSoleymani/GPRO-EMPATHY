# GPRO-EMPATHY: Group Relative Policy Optimization for Empathetic Response Generation

## Abstract

This report presents GPRO-EMPATHY, a novel approach for training empathetic conversational AI systems using Group Relative Policy Optimization (GRPO). The system combines multiple reward functions - semantic similarity and empathy prediction - to train a Llama 3.1-8B model for generating contextually appropriate and emotionally resonant responses. The approach leverages the WASSA Empathy dataset and employs a fine-tuned RoBERTa model for empathy assessment, achieving improved empathetic response generation through reinforcement learning from human feedback principles.

## 1. Introduction

Empathetic response generation is a crucial challenge in conversational AI, requiring systems to understand emotional context and respond appropriately to users' emotional states. Traditional supervised fine-tuning approaches often fall short in generating responses that demonstrate genuine empathy and emotional intelligence.

This work introduces GPRO-EMPATHY, which addresses these limitations by:
- Employing Group Relative Policy Optimization (GRPO) for training empathetic responses
- Using dual reward functions for comprehensive empathy evaluation
- Implementing structured reasoning through XML-formatted responses
- Leveraging LoRA (Low-Rank Adaptation) for efficient fine-tuning

## 2. Related Work

### 2.1 Empathy in Conversational AI
Previous work has demonstrated the importance of empathy in human-computer interaction. The WASSA Shared Task on Empathy Detection and Emotion Classification (Barriere et al., 2022) established benchmarks for measuring empathy in conversational contexts, providing structured datasets and evaluation metrics for empathetic AI systems.

### 2.2 Reinforcement Learning from Human Feedback
Recent advances in RLHF have shown significant improvements in language model alignment. Group Relative Policy Optimization represents a more efficient alternative to traditional PPO-based approaches, offering better sample efficiency and stability in training (Ahmadian et al., 2024).

### 2.3 Multi-Reward Training
The integration of multiple reward signals has proven effective in training language models for complex objectives, allowing systems to optimize for multiple desirable characteristics simultaneously.

## 3. Methodology

### 3.1 Architecture Overview

The GPRO-EMPATHY system consists of several key components:

1. **Base Model**: Meta-Llama-3.1-8B-Instruct
2. **Fine-tuning**: LoRA with rank 32 for efficient adaptation
3. **Training Algorithm**: Group Relative Policy Optimization (GRPO)
4. **Reward Functions**: Dual-reward system combining semantic similarity and empathy prediction

### 3.2 Dataset

**WASSA Empathy Dataset** (`miladsolo/wassa-conv-turn-empathy`):
- Conversational utterances with empathy labels (0-5 scale)
- Text field contains user messages requiring empathetic responses
- Labels used for validation, not direct supervision

### 3.3 System Prompt Design

The system employs a carefully crafted prompt that guides the model to:
```xml
<reasoning>
- User's main concern: [identify what they're expressing]
- Emotional content: [what emotions/feelings are present]
- Intensity level: [0=neutral, 1-2=mild, 3-4=moderate, 5=high emotional intensity]
- Response approach: [how to respond empathetically]
</reasoning>
<answer>
[Your empathetic response here - 1-2 sentences maximum]
</answer>
```

This structured approach enables:
- Explicit emotional analysis
- Graduated empathy responses based on intensity
- Separation of reasoning from response generation
- Consistent output formatting

### 3.4 Reward Functions

#### 3.4.1 Semantic Similarity Reward
- **Model**: `cross-encoder/stsb-roberta-large`
- **Purpose**: Measures relevance and contextual appropriateness
- **Input**: User utterance vs. generated response (answer section only)
- **Output**: Calibrated score [0,1]

#### 3.4.2 Empathy Prediction Reward
- **Model**: `miladsolo/roberta-lora-wassa-empathy` (fine-tuned RoBERTa)
- **Purpose**: Evaluates empathetic quality of responses
- **Training**: Based on WASSA empathy dataset using fine-tuning-roberta.ipynb methodology
- **Output**: Empathy logit converted to [0,1] range

#### 3.4.3 Reward Calibration
Both rewards undergo batch calibration using temperature scaling (T=0.6) to normalize distributions and improve training stability.

### 3.5 Training Configuration

```yaml
Model Configuration:
- Base Model: meta-llama/meta-Llama-3.1-8B-Instruct
- Max Sequence Length: 1024 tokens
- LoRA Rank: 32
- 4-bit Quantization: Enabled
- GPU Memory Utilization: 0.6

Training Parameters:
- Learning Rate: 1e-3
- Optimizer: paged_adamw_8bit
- Batch Size: 1 per device
- Gradient Accumulation: 1 step
- Number of Generations: 3
- Max Training Steps: 250
- Save Steps: 250
- Max Prompt Length: 512 tokens
```

### 3.6 GRPO Training Process

1. **Prompt Processing**: User messages formatted with system instructions
2. **Generation**: Model generates multiple response candidates (N=3)
3. **Reward Calculation**: Both reward functions score the answer portions
4. **Policy Update**: GRPO uses reward signals to update model parameters
5. **Iteration**: Process repeats for specified training steps

## 4. Implementation Details

### 4.1 Model Architecture
- **Base**: Llama 3.1-8B with instruction tuning
- **Adaptation**: LoRA targeting all attention layers (q_proj, k_proj, v_proj, o_proj) and MLP layers (gate_proj, up_proj, down_proj)
- **Memory Optimization**: Unsloth integration for efficient training
- **Inference**: vLLM backend for fast generation

### 4.2 Reward Function Implementation

The reward functions process only the `<answer>` section of generated responses, ensuring evaluation focuses on user-facing content rather than internal reasoning:

```python
def _extract_answer_text(reply_text: str) -> str:
    """Extract only the answer section for reward calculation."""
    ans = _extract_text_between(reply_text, _ANSWER_RE, fallback=reply_text)
    return ans.strip()
```

### 4.3 Training Pipeline
1. **Dataset Loading**: WASSA empathy data processed into chat format
2. **Model Initialization**: Base model loaded with LoRA adaptation
3. **GRPO Setup**: Training configuration with dual reward functions
4. **Training Loop**: Iterative improvement through reward optimization
5. **Model Saving**: LoRA adapter preservation for deployment

## 5. Evaluation Framework

### 5.1 Reward Metrics
- **Semantic Similarity**: Measures contextual relevance and appropriateness
- **Empathy Score**: Quantifies emotional resonance and supportiveness
- **Combined Score**: Weighted average of both metrics

### 5.2 Response Quality Assessment
Generated responses are evaluated on:
- Emotional appropriateness to user's state
- Contextual relevance to conversation
- Empathetic language and tone
- Structured reasoning quality

## 6. Results and Analysis

### 6.1 Training Observations
- Model successfully learns to generate XML-formatted responses
- Empathy reasoning shows graduated responses based on emotional intensity
- Reward functions effectively guide model toward more empathetic outputs
- LoRA adaptation enables efficient training with limited computational resources

### 6.2 Response Quality Examples

**Low Empathy Input (Level 1-2)**:
```
Input: "what did you think about this article"
Response: "I understand you're asking about this. What aspects of this situation are most important to you?"
```

**High Empathy Input (Level 4-5)**:
```
Input: "I think it's super sad... they seem to never catch a break, always struggling"
Response: "The depth of care you're expressing is really touching. I can feel how much their struggles weigh on your heart - carrying that level of empathy for others can be both meaningful and emotionally heavy."
```

## 7. Technical Contributions

### 7.1 Novel Aspects
1. **Dual-Reward GRPO**: First application of GRPO with semantic + empathy rewards
2. **Structured Empathy Generation**: XML-formatted responses with explicit reasoning
3. **Graduated Empathy**: Intensity-aware response generation
4. **Efficient Training**: LoRA + Unsloth integration for resource optimization

### 7.2 System Architecture
- Modular reward function design enabling easy extension
- Clean separation of reasoning and response generation
- Robust error handling and type conversion
- Comprehensive configuration management

## 8. Future Work

### 8.1 Potential Improvements
- Multi-turn conversation empathy tracking
- Additional reward functions (safety, helpfulness)
- Larger model architectures (70B+ parameters)
- Human evaluation studies
- Cross-cultural empathy adaptation

### 8.2 Research Directions
- Long-term empathy consistency in conversations
- Personalized empathy adaptation
- Multi-modal empathy (text + voice/visual cues)
- Empathy transfer across domains

## 9. Conclusion

GPRO-EMPATHY demonstrates the effectiveness of combining Group Relative Policy Optimization with specialized reward functions for empathetic response generation. The system successfully trains language models to generate contextually appropriate, emotionally resonant responses through structured reasoning and multi-objective optimization.

Key achievements:
- Successful integration of GRPO with empathy-specific rewards
- Development of structured empathy reasoning framework
- Efficient training pipeline using modern optimization techniques
- Demonstrated improvement in empathetic response quality

The work opens new avenues for developing more emotionally intelligent AI systems and provides a foundation for future research in computational empathy.

## 10. References

- Barriere, V., Balazs, J. A., Karimi, A., Schrader, L., Amin, L., Riemann, D., ... & Klakow, D. (2022). WASSA 2022 shared task on empathy detection, emotion classification and personality detection. In *Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis* (pp. 249-264).

- Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J., Üstün, A., & Hooker, S. (2024). Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. *arXiv preprint arXiv:2406.19071*.

## Appendix A: Configuration Files

### A.1 Training Configuration
```yaml
# GPRO Empathy Training Configuration
model:
  name: "meta-llama/meta-Llama-3.1-8B-Instruct"
  max_seq_length: 1024
  lora_rank: 32
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.6

training:
  learning_rate: 1e-3
  adam_beta1: 0.9
  adam_beta2: 0.99
  weight_decay: 0.1
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  optim: "paged_adamw_8bit"
  logging_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  num_generations: 3
  max_steps: 250
  save_steps: 250
  max_grad_norm: 0.1
  output_dir: "outputs"
  max_prompt_length: 512

rewards:
  use_semantic_similarity: true
  use_empathy_model: true
  empathy_model_repo: "miladsolo/roberta-lora-wassa-empathy"
  calibration_temperature: 0.6
```

## Appendix B: System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  System Prompt   │───▶│ Llama 3.1-8B    │
│                 │    │   + Template     │    │   + LoRA        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ XML Response    │
                                               │ Generation      │
                                               └─────────────────┘
                                                        │
                                               ┌────────┴────────┐
                                               ▼                 ▼
                                    ┌─────────────────┐ ┌─────────────────┐
                                    │   Reasoning     │ │     Answer      │
                                    │   Section       │ │    Section      │
                                    └─────────────────┘ └─────────────────┘
                                                                 │
                                                                 ▼
                                                      ┌─────────────────┐
                                                      │ Reward Functions │
                                                      │ - Semantic      │
                                                      │ - Empathy       │
                                                      └─────────────────┘
                                                                 │
                                                                 ▼
                                                      ┌─────────────────┐
                                                      │ GRPO Training   │
                                                      │ Update          │
                                                      └─────────────────┘
```

---

*This report documents the GPRO-EMPATHY system developed for empathetic conversational AI using modern reinforcement learning techniques and specialized reward functions for emotional intelligence.*