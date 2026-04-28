import re
from typing import Dict, List, Optional

from datasets import load_dataset


_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _normalize_numeric_text(text: str) -> str:
    cleaned = text.replace(",", "").replace("$", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(".")
    return cleaned


def extract_gsm8k_answer(answer_text: str) -> str:
    text = answer_text.strip()

    if "####" in text:
        final = text.split("####")[-1]
        return _normalize_numeric_text(final)

    matches = _NUMBER_PATTERN.findall(text.replace(",", "").replace("$", ""))
    if matches:
        return _normalize_numeric_text(matches[-1])

    return _normalize_numeric_text(text)


def load_gsm8k(split: str = "train", limit: Optional[int] = None) -> List[Dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    records: List[Dict[str, str]] = []
    for idx, item in enumerate(dataset):
        if limit is not None and idx >= limit:
            break

        final_answer = extract_gsm8k_answer(item["answer"])
        records.append(
            {
                "question": item["question"].strip() + "\nAnswer:",
                "answer": final_answer,
            }
        )

    return records
