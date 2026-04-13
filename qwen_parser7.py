import json
import re
from typing import Any, Callable

import dirtyjson

from qwen_logging import log_event, log_retry
from qwen_prompts import SYSTEM_TEXT_INTERPRET_V2, build_interpret_prompt_v2


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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return dirtyjson.loads(raw)


def validate_interpret_output(data: dict[str, Any], mode: str) -> dict[str, Any]:
    required_keys = {
        "status",
        "target_description",
        "source_description",
        "destination_description",
        "input_text",
        "generate_random_text",
        "random_length",
        "comment",
    }
    if set(data.keys()) != required_keys:
        raise ValueError("unexpected interpret keys")

    status = data["status"]
    if status not in {"ok", "ambiguous"}:
        raise ValueError("interpret status must be 'ok' or 'ambiguous'")

    for key in ("target_description", "source_description", "destination_description"):
        if not (data[key] is None or isinstance(data[key], str)):
            raise ValueError(f"{key} must be string or null")
    if not isinstance(data["input_text"], str):
        raise ValueError("input_text must be string")
    if not isinstance(data["generate_random_text"], bool):
        raise ValueError("generate_random_text must be boolean")
    if not (data["random_length"] is None or isinstance(data["random_length"], int)):
        raise ValueError("random_length must be integer or null")
    if not isinstance(data["comment"], str):
        raise ValueError("comment must be string")
    if isinstance(data["random_length"], int) and data["random_length"] <= 0:
        raise ValueError("random_length must be positive")

    validated = {
        "status": status,
        "target_description": clean_single_line_text(data["target_description"] or "") or None,
        "source_description": clean_single_line_text(data["source_description"] or "") or None,
        "destination_description": clean_single_line_text(data["destination_description"] or "") or None,
        "input_text": clean_single_line_text(data["input_text"]),
        "generate_random_text": data["generate_random_text"],
        "random_length": data["random_length"],
        "comment": data["comment"],
    }

    if status == "ambiguous":
        return validated

    if mode == "drag":
        if not validated["source_description"] or not validated["destination_description"]:
            raise ValueError("drag requires both source_description and destination_description")
        if validated["target_description"] is not None:
            raise ValueError("target_description must be null for drag")
        if validated["input_text"] or validated["generate_random_text"] or validated["random_length"] is not None:
            raise ValueError("drag must not include input fields")
        return validated

    if mode == "input":
        if not validated["target_description"]:
            raise ValueError("input requires target_description")
        if validated["generate_random_text"] and validated["input_text"]:
            raise ValueError("input_text must be empty when generate_random_text is true")
        if not validated["generate_random_text"] and not validated["input_text"]:
            raise ValueError("input requires literal text or generate_random_text=true")
        if validated["source_description"] is not None or validated["destination_description"] is not None:
            raise ValueError("input must not include drag fields")
        return validated

    if mode in {"yes_no", "point", "value", "multi_value"}:
        if not validated["target_description"]:
            raise ValueError(f"{mode} requires target_description")
        if validated["source_description"] is not None or validated["destination_description"] is not None:
            raise ValueError(f"{mode} must not include drag fields")
        if validated["input_text"] or validated["generate_random_text"] or validated["random_length"] is not None:
            raise ValueError(f"{mode} must not include input fields")
        return validated

    raise ValueError(f"Unsupported mode: {mode}")


def interpret_command(
    question: str,
    mode: str,
    run_text_model: Callable[[list[dict[str, Any]]], str],
    max_retries: int = 2,
) -> dict[str, Any]:
    requested_input_length = extract_requested_input_length(question) if mode == "input" else None
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for attempt in range(1, max_retries + 1):
        prompt = build_interpret_prompt_v2(mode, question, requested_input_length)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )

        log_event(
            "interpret.request",
            f"attempt {attempt}/{max_retries}",
            {
                "mode": mode,
                "question": question,
                "prompt": prompt,
            },
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_TEXT_INTERPRET_V2}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        try:
            raw_text = run_text_model(messages)
            last_raw_text = raw_text
            log_event("interpret.raw_output", payload=raw_text)
            data = extract_json(raw_text)
            validated = validate_interpret_output(data, mode)
            log_event("interpret.result", payload=validated)
            return validated
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("interpret", attempt, max_retries, exc, last_raw_text)

    raise RuntimeError(
        f"Interpret failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )
