import json
import re
from typing import Any

import dirtyjson

def clean_single_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_requested_input_length(question: str) -> int | None:
    patterns = [
        r"length\s+of\s+(\d+)\s*(?:symbols?|chars?|characters?)",
        r"(\d+)\s*(?:symbols?|chars?|characters?)",
    ]
    q = question.lower()
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"JSON object not found in model output: {text!r}")

    raw = text[start : end + 1]
    raw = raw.replace("'", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return dirtyjson.loads(raw)
