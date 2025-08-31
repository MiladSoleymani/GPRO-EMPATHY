# Empathy Model Training Documentation

## Overview

This document describes the training process for the RoBERTa-based empathy prediction model used as a reward function in the GPRO-EMPATHY system.

## Training Process (Based on fine-tuning-roberta.ipynb)

### 1. Model Architecture
- **Base Model**: `roberta-base`
- **Task**: Multi-class classification (3 outputs)
- **Classes**: Emotion, EmotionalPolarity, Empathy
- **Adaptation**: LoRA fine-tuning for efficiency

### 2. Data Processing

#### Text Normalization
```python
def normalize_text_roberta(text: str) -> str:
    """Comprehensive text cleaning for RoBERTa training"""
    - Unicode normalization (NFKC)
    - URL replacement with <url> tokens
    - Email replacement with <email> tokens  
    - User mention normalization (@USER)
    - Hashtag detagging (keep content, remove #)
    - Control character removal
    - Whitespace collapsing
```

#### Dataset Processing
- **Source**: WASSA Empathy Dataset
- **Text Column**: Automatic detection from multiple candidates
- **Labels**: (Emotion, EmotionalPolarity, Empathy) as float32
- **Tokenization**: RoBERTa tokenizer with max_length=256
- **Split**: 90% train, 10% validation (seed=42)

### 3. Training Configuration

```python
Model Configuration:
- Base: roberta-base
- Max Length: 256 tokens
- Label Columns: ["Emotion", "EmotionalPolarity", "Empathy"]
- Batch Processing: Enabled for efficiency

Preprocessing Pipeline:
- Text cleaning via normalize_text_roberta
- Tokenization with padding="max_length"
- Label stacking as numpy arrays
- Format conversion to PyTorch tensors
```

### 4. Model Integration

#### In GRPO-EMPATHY System
The trained RoBERTa model serves as the empathy reward function:

```python
class EmpathyModelReward:
    def __init__(self, model_repo="miladsolo/roberta-lora-wassa-empathy"):
        self._device = "cuda" if available else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_repo)
        self._cls = AutoModelForSequenceClassification.from_pretrained(model_repo)
    
    def predict(self, texts, max_len=256):
        # Tokenization and encoding
        # GPU processing with torch.no_grad()
        # Return 3-dimensional predictions
    
    def __call__(self, prompts, completions, **kwargs):
        # Extract answer text from completions
        # Predict empathy scores
        # Apply calibration
        # Return [0,1] scaled rewards
```

#### Reward Calculation Process
1. **Input**: Generated model responses (answer sections only)
2. **Processing**: Text normalization and tokenization
3. **Prediction**: 3-class logits from fine-tuned RoBERTa
4. **Extraction**: Empathy dimension (index 2) from predictions
5. **Calibration**: Temperature-based scaling for [0,1] range

### 5. Performance Characteristics

#### Model Specifications
- **Parameters**: ~125M (RoBERTa-base with LoRA)
- **Input Length**: Up to 256 tokens
- **Processing Speed**: Batch processing for efficiency
- **Memory Usage**: Optimized for GPU deployment

#### Output Format
```python
{
    "Emotion": float,           # Emotional content score
    "EmotionalPolarity": float, # Positive/negative emotion
    "Empathy": float           # Empathy level (used for reward)
}
```

### 6. Integration with GRPO Training

#### Reward Function Role
- **Purpose**: Evaluate empathetic quality of generated responses
- **Input**: Model-generated answer text (XML extracted)
- **Output**: Calibrated empathy scores [0,1]
- **Frequency**: Called for each generation during GRPO training

#### Calibration Process
```python
def _batch_calibrate(raw_scores, temperature=0.6):
    # Z-score normalization
    # Temperature scaling
    # Sigmoid transformation
    # Return [0,1] calibrated scores
```

### 7. Quality Assurance

#### Text Processing Robustness
- Handles various input formats (strings, dicts, nested structures)
- Graceful fallback for empty/invalid inputs
- Comprehensive text cleaning pipeline
- Unicode and encoding safety

#### Error Handling
- GPU/CPU device management
- Batch processing with error recovery
- Fallback predictions for failed cases
- Memory management for large batches

### 8. Model Deployment

#### HuggingFace Hub Integration
- Model repository: `miladsolo/roberta-lora-wassa-empathy`
- Automatic downloading and caching
- Version control and reproducibility
- Easy integration in training pipeline

#### Runtime Optimization
- Model loaded once, reused across training
- Efficient tokenization with padding strategies
- GPU memory management
- Batch processing for throughput

---

## Training Data Statistics

Based on the WASSA empathy dataset:
- **Total Samples**: Variable (processed during training)
- **Text Quality**: High-quality conversational utterances  
- **Label Distribution**: Balanced across empathy levels
- **Preprocessing**: Comprehensive text normalization

## Usage in GRPO-EMPATHY

The empathy model serves as a crucial component of the reward system:

1. **Training Phase**: Evaluates generated responses for empathetic content
2. **Reward Signal**: Provides gradient information for policy optimization
3. **Quality Control**: Ensures responses demonstrate appropriate empathy levels
4. **Calibration**: Normalized scores enable stable GRPO training

This empathy model training process ensures robust, reliable evaluation of empathetic content in the GRPO-EMPATHY system.