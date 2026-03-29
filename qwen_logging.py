import json
import os
import sys
from typing import Any


VERBOSE_LOGGING_ENABLED = os.environ.get("QWEN_VERBOSE_LOGGING", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
MAX_LOG_TEXT_CHARS = int(os.environ.get("QWEN_MAX_LOG_TEXT_CHARS", "4000"))


def _truncate_text(text: str, max_chars: int = MAX_LOG_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _format_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return _truncate_text(payload)
    try:
        return _truncate_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        return _truncate_text(repr(payload))


def log_event(label: str, message: str | None = None, payload: Any | None = None) -> None:
    if not VERBOSE_LOGGING_ENABLED:
        return

    header = f"[qwen:{label}]"
    if message:
        print(f"{header} {message}", file=sys.stderr, flush=True)
    else:
        print(header, file=sys.stderr, flush=True)

    if payload is not None:
        print(_format_payload(payload), file=sys.stderr, flush=True)


def log_retry(
    label: str,
    attempt: int,
    max_retries: int,
    error: Exception,
    raw_text: str | None = None,
) -> None:
    if not VERBOSE_LOGGING_ENABLED:
        return

    print(
        f"[qwen:{label}] retry {attempt}/{max_retries} because: {error}",
        file=sys.stderr,
        flush=True,
    )
    if raw_text is not None:
        print(_format_payload(raw_text), file=sys.stderr, flush=True)
