import re
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE
)
_USER_SPAN_RE = re.compile(
    r"<\|user\|>\s*(.*?)\s*</s>", flags=re.DOTALL | re.IGNORECASE
)


def _extract_text_between(s: str, pattern: re.Pattern, fallback: str = "") -> str:
    m = pattern.search(s or "")
    return m.group(1).strip() if m else (fallback or "").strip()


def _extract_utterance_from_prompt(prompt_text: str) -> str:
    """Pull just the user utterance from your chat template."""
    text = _extract_text_between(prompt_text, _USER_SPAN_RE, fallback=prompt_text)
    return re.sub(r"</?[^>]+>", "", text).strip()


def _extract_answer_text(reply_text: str) -> str:
    """If XML is present, score only the <answer>…</answer> body."""
    if reply_text:
        print(f"🔍 Extracting answer from: {reply_text[:100]}...")
    ans = _extract_text_between(reply_text or "", _ANSWER_RE, fallback=reply_text or "")
    extracted = ans.strip()
    print(f"  → Extracted answer: {extracted[:50]}{'...' if len(extracted) > 50 else ''}")
    return extracted


def _flatten_completions(completions) -> list[str]:
    """Handle various TRL completion shapes and return list[str]."""
    print(f"🔍 _flatten_completions called with {len(completions) if completions else 0} completions")
    print(f"🔍 Completion types: {[type(c).__name__ for c in (completions or [])]}")
    
    out = []
    for i, c in enumerate(completions or []):
        if isinstance(c, str):
            print(f"  [{i}] String completion: {len(c)} chars")
            out.append(c)
        elif isinstance(c, dict) and "content" in c:
            content = c["content"]
            print(f"  [{i}] Dict completion: {len(content)} chars")
            out.append(content)
        elif isinstance(c, (list, tuple)) and len(c) > 0:
            first = c[0]
            if isinstance(first, dict) and "content" in first:
                content = first["content"]
                print(f"  [{i}] Nested dict completion: {len(content)} chars")
                out.append(content)
            elif (
                isinstance(first, (list, tuple))
                and len(first) > 0
                and isinstance(first[0], dict)
                and "content" in first[0]
            ):
                content = first[0]["content"]
                print(f"  [{i}] Deep nested completion: {len(content)} chars")
                out.append(content)
            else:
                print(f"  [{i}] Converted to string: {str(c)[:50]}...")
                out.append(str(c))
        else:
            print(f"  [{i}] Empty/invalid completion, using empty string")
            out.append("")
    
    print(f"🎯 _flatten_completions returning {len(out)} strings")
    return out


def _batch_calibrate(raw_scores: np.ndarray, temperature: float = 0.6) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)
    if raw_scores.size == 0:
        return raw_scores
    raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=1.0, neginf=0.0)
    mu, sigma = raw_scores.mean(), raw_scores.std()
    if sigma < 1e-6:
        return np.clip(1.0 / (1.0 + np.exp(-(raw_scores - mu))), 0.0, 1.0)
    z = (raw_scores - mu) / sigma
    t = max(1e-4, float(temperature))
    return 1.0 / (1.0 + np.exp(-z / t))


class SemanticSimilarityReward:
    def __init__(self):
        self._ce = CrossEncoder(
            "cross-encoder/stsb-roberta-large",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        """
        Reward = calibrated semantic similarity between:
          source = user utterance extracted from the prompt
          reply  = model's <answer> text (or full reply if no XML)
        Returns floats in [0,1].
        """
        print(f"🔍 SemanticSimilarityReward processing {len(prompts)} prompts")
        user_msgs = [p[-1]["content"] for p in prompts]
        sources = [_extract_utterance_from_prompt(m) for m in user_msgs]
        print(f"📝 Extracted {len(sources)} user sources")

        print(f"🔍 Processing {len(completions) if completions else 0} completions")
        reply_texts = _flatten_completions(completions)
        print(f"📝 Flattened to {len(reply_texts)} reply texts")
        replies = [_extract_answer_text(t) for t in reply_texts]
        print(f"📝 Extracted {len(replies)} answer texts")

        pairs = []
        for s, r in zip(sources, replies):
            s = (s or "").strip()
            r = (r or "").strip()
            pairs.append((s if s else "x", r if r else "x"))

        # print("SemanticSimilarityReward: ", pairs)
        try:
            raw = np.array(self._ce.predict(pairs, batch_size=64), dtype=float)
            print(f"✅ Semantic reward computed: {len(raw)} scores for {len(pairs)} pairs")
        except Exception as e:
            print(f"⚠️ Semantic reward failed: {e}")
            raw = np.zeros(len(pairs), dtype=float)

        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
        if raw.size and raw.max() > 1.25:
            raw = raw / 5.0
        raw = np.clip(raw, 0.0, 1.0)

        cal = _batch_calibrate(raw, temperature=0.6)
        return cal.tolist()


class EmpathyModelReward:
    def __init__(self, model_repo: str = "miladsolo/roberta-lora-wassa-empathy"):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_repo)
        self._cls = AutoModelForSequenceClassification.from_pretrained(model_repo)
        self._cls.eval().to(self._device)

    def predict(self, texts, max_len=256):
        # print("EmpathyModelReward: ", texts)
        try:
            enc = self._tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._cls(**enc).logits
            arr = logits.detach().cpu().numpy()
            print(f"✅ Empathy reward computed: {len(arr)} scores for {len(texts)} texts")
        except Exception as e:
            print(f"⚠️ Empathy reward failed: {e}")
            arr = np.zeros((len(texts), 3), dtype=float)
        return [
            {
                "Emotion": float(a[0]),
                "EmotionalPolarity": float(a[1]),
                "Empathy": float(a[2]),
            }
            for a in arr
        ]

    def __call__(self, prompts=None, completions=None, **kwargs) -> list[float]:
        """
        Reward = model-predicted Empathy logit for the assistant's reply (higher is better).
        Uses miladsolo/roberta-lora-wassa-empathy via `predict()`. Calibrated to [0,1].
        """
        print(f"🔍 EmpathyModelReward processing {len(completions or [])} completions")
        reply_texts = _flatten_completions(completions or [])
        print(f"📝 Flattened to {len(reply_texts)} reply texts")
        answers = [_extract_answer_text(t) for t in reply_texts]
        print(f"📝 Extracted {len(answers)} answer texts")
        safe_inputs = [a if a else " " for a in answers]
        print(f"📝 Created {len(safe_inputs)} safe inputs for empathy model")

        preds = self.predict(safe_inputs)
        raw = np.array([p.get("Empathy", 0.0) for p in preds], dtype=float)

        cal = _batch_calibrate(raw, temperature=0.6)
        return np.clip(cal, 0.0, 1.0).tolist()


def semantic_sts_reward(prompts, completions, **kwargs) -> list[float]:
    """Convenience function for semantic similarity reward."""
    print(f"🔍 Semantic reward called with {len(prompts)} prompts, {len(completions)} completions")
    reward_fn = SemanticSimilarityReward()
    results = reward_fn(prompts, completions, **kwargs)
    print(f"🎯 Semantic reward returned {len(results)} scores")
    return results


def empathy_model_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Convenience function for empathy model reward."""
    print(f"🔍 Empathy reward called with {len(prompts or [])} prompts, {len(completions or [])} completions")
    reward_fn = EmpathyModelReward()
    results = reward_fn(prompts, completions, **kwargs)
    print(f"💝 Empathy reward returned {len(results)} scores")
    return results
