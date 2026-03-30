import json
import re
from typing import Any, Callable

from qwen_logging import log_event, log_retry
from qwen_prompts import SYSTEM_TEXT_PARSE, build_parse_prompt


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
    return json.loads(raw)


def validate_parser_output(data: dict[str, Any], mode: str) -> dict[str, Any]:
    status = data.get("status")
    if status not in {"ok", "ambiguous"}:
        raise ValueError("parser status must be 'ok' or 'ambiguous'")

    if mode in {"yes_no", "point"}:
        if set(data.keys()) != {"status", "target_description", "comment"}:
            raise ValueError("unexpected parser keys for yes_no/point")
        target_description = data["target_description"]
        comment = data["comment"]
        if not (target_description is None or isinstance(target_description, str)):
            raise ValueError("target_description must be string or null")
        if not isinstance(comment, str):
            raise ValueError("comment must be string")
        return {
            "status": status,
            "target_description": clean_single_line_text(target_description or "") or None,
            "comment": comment,
        }

    if mode == "input":
        required_keys = {
            "status",
            "target_description",
            "input_text",
            "generate_random_text",
            "random_length",
            "comment",
        }
        if set(data.keys()) != required_keys:
            raise ValueError("unexpected parser keys for input")
        if not (data["target_description"] is None or isinstance(data["target_description"], str)):
            raise ValueError("target_description must be string or null")
        if not isinstance(data["input_text"], str):
            raise ValueError("input_text must be string")
        if not isinstance(data["generate_random_text"], bool):
            raise ValueError("generate_random_text must be boolean")
        if not (data["random_length"] is None or isinstance(data["random_length"], int)):
            raise ValueError("random_length must be integer or null")
        if not isinstance(data["comment"], str):
            raise ValueError("comment must be string")
        return {
            "status": status,
            "target_description": clean_single_line_text(data["target_description"] or "") or None,
            "input_text": clean_single_line_text(data["input_text"]),
            "generate_random_text": data["generate_random_text"],
            "random_length": data["random_length"],
            "comment": data["comment"],
        }

    if mode == "drag":
        if set(data.keys()) != {"status", "source_description", "destination_description", "comment"}:
            raise ValueError("unexpected parser keys for drag")
        if not (data["source_description"] is None or isinstance(data["source_description"], str)):
            raise ValueError("source_description must be string or null")
        if not (data["destination_description"] is None or isinstance(data["destination_description"], str)):
            raise ValueError("destination_description must be string or null")
        if not isinstance(data["comment"], str):
            raise ValueError("comment must be string")
        return {
            "status": status,
            "source_description": clean_single_line_text(data["source_description"] or "") or None,
            "destination_description": clean_single_line_text(data["destination_description"] or "") or None,
            "comment": data["comment"],
        }

    raise ValueError("Unsupported mode")


def call_parser(
    question: str,
    mode: str,
    run_text_model: Callable[[list[dict[str, Any]]], str],
    max_retries: int = 3,
) -> dict[str, Any]:
    requested_input_length = extract_requested_input_length(question) if mode == "input" else None
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for _attempt in range(1, max_retries + 1):
        user_prompt = build_parse_prompt(mode, question, requested_input_length)
        if retry_feedback:
            user_prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )

        log_event(
            "parser.request",
            f"attempt {_attempt}/{max_retries}",
            {
                "mode": mode,
                "question": question,
                "prompt": user_prompt,
            },
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_TEXT_PARSE}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        try:
            raw_text = run_text_model(messages)
            last_raw_text = raw_text
            log_event("parser.raw_output", payload=raw_text)
            data = extract_json(raw_text)
            log_event("parser.json", payload=data)
            validated = validate_parser_output(data, mode)
            log_event("parser.result", payload=validated)
            return validated
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("parser", _attempt, max_retries, exc, last_raw_text)

    raise RuntimeError(
        f"Parser failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )
