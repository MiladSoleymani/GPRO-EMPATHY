from typing import Optional, Sequence
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


SYSTEM_PROMPT = """
<|system|>
You are a friendly, trauma-informed assistant. You receive prior dialogue turns marked with <|user|> and <|assistant|>.
Continue the conversation as <|assistant|> with an empathetic, concise reply that:
- reflects the user's experience in new words,
- validates feelings,
- optionally asks ONE gentle, forward-moving question,
- stays within 1–2 sentences,
- avoids emojis, lists, quotes, and clinical/lecturing tone,
- maintains continuity and never repeats the prompt.

Output EXACTLY this XML wrapper:

<reasoning>
- Briefly identify the user's main concern.
- Name the likely emotion and a rough 0–5 intensity.
- Plan one supportive move (reflection / validation / next step).
</reasoning>
<answer>
(Your final 1–2 sentence reply here.)
</answer>
</s>
""".strip()


_TEXT_COL_CANDIDATES = [
    "utterance",
    "Utterance",
    "text",
    "Text",
    "content",
    "message",
    "sentence",
    "prompt",
]


def _pick_text_col(cols: Sequence[str]) -> str:
    for c in _TEXT_COL_CANDIDATES:
        if c in cols:
            return c
    return cols[0]


def _to_int_0_5(x) -> Optional[int]:
    try:
        v = float(x)
    except Exception:
        return None
    v = round(v)
    v = max(0, min(5, int(v)))
    return v


def _mk_instruction(utterance: str) -> str:
    return (
        "Here is the dialogue so far. Continue as <|assistant|>.\n\n"
        f"<|user|>\n{utterance}\n</s>\n"
    )


def load_wassa_empathy(split: Optional[str] = None) -> Dataset:
    """Load and format WASSA empathy dataset for GPRO training."""
    raw = (
        load_dataset("miladsolo/wassa-conv-turn-empathy")
        if split is None
        else load_dataset("miladsolo/wassa-conv-turn-empathy", split=split)
    )
    ds = (
        concatenate_datasets([raw[k] for k in raw.keys()])
        if isinstance(raw, DatasetDict)
        else raw
    )

    cols = ds.column_names
    text_col = _pick_text_col(cols)

    def _map(ex):
        text = (ex.get(text_col) or "").strip()
        if not text:
            return {"prompt": None, "answer": None}
        y = _to_int_0_5(ex.get("Empathy", None))
        if y is None:
            return {"prompt": None, "answer": None}
        user_msg = _mk_instruction(text)
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "answer": str(y),
        }

    ds = ds.map(
        _map, remove_columns=[c for c in cols if c not in (text_col, "Empathy")]
    )
    ds = ds.filter(
        lambda ex: isinstance(ex["prompt"], list)
        and isinstance(ex["answer"], str)
        and len(ex["answer"]) > 0
    )

    print(f"Prepared {len(ds)} WASSA empathy examples.")

    # Quick preview
    for i in range(min(3, len(ds))):
        ex = ds[i]
        print(f"\n--- sample {i} ---")
        print("Q:", ex["prompt"][-1]["content"][:300])
        print("A:", ex["answer"])

    return ds


def get_system_prompt() -> str:
    """Get the system prompt used for training."""
    return SYSTEM_PROMPT
