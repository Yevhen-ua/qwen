import json
import os
import random
import re
import string
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from raw_answer_point import draw

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")

# Prefer FlashAttention 2 when available, otherwise fall back safely.
ATTN_IMPLEMENTATION = "sdpa"
try:
    import flash_attn  # noqa: F401

    ATTN_IMPLEMENTATION = "flash_attention_2"
except Exception:
    pass

# Image resizing control for Qwen3-VL.
IMAGE_MIN_PIXELS = 256 * 256
IMAGE_MAX_PIXELS = 1200 * 1200

MAX_NEW_TOKENS = 120
MAX_REASON_WORDS = 12
NORMALIZED_COORD_MAX = 1000
DEFAULT_RANDOM_TEXT_LENGTH = 12

if torch.cuda.is_available():
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
else:
    DTYPE = torch.float32

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map="auto",
    attn_implementation=ATTN_IMPLEMENTATION,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

SYSTEM_TEXT_PARSE = (
    "You convert a user's natural-language UI command into a strict internal task. "
    "Return exactly one valid JSON object and nothing else. "
    "Preserve only attributes that the user explicitly requested or clearly implied. "
    "Keep visible text snippets exactly as written, including Cyrillic, Latin letters, digits, and punctuation. "
    "Do not invent missing color, type, state, label, placeholder, or destination details. "
    "If the command is ambiguous, return status='ambiguous' instead of guessing."
)

SYSTEM_TEXT_VISION = (
    "You analyze exactly one screenshot of a website or web application. "
    "Return exactly one valid JSON object and nothing else. "
    "Use only information visibly present in the screenshot. "
    "The provided internal task is the source of truth. "
    "Every non-null attribute in the internal task must match. "
    "If there is no exact visible match, return not_found. "
    "If several candidates still match and the target is not uniquely determined, return ambiguous."
)


def normalize_mode(mode: str) -> str:
    value = mode.strip().lower()
    aliases = {
        "y_n": "yes_no",
        "yes/no": "yes_no",
        "yes_no": "yes_no",
        "yn": "yes_no",
        "point": "point",
        "input": "input",
        "drag": "drag",
    }
    if value not in aliases:
        raise ValueError("mode must be one of: yes_no, y_n, point, input, drag")
    return aliases[value]


def limit_reason_words(text: str, max_words: int = MAX_REASON_WORDS) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    words = text.split()
    return " ".join(words[:max_words])


def clean_single_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_requested_input_length(question: str) -> int | None:
    patterns = [
        r"length\s+of\s+(\d+)\s*(?:symbols?|chars?|characters?)",
        r"(\d+)\s*(?:symbols?|chars?|characters?)",
        r"довжин[аою]\s+(\d+)\s*(?:символ(?:ів|и)?|знаків?)",
        r"(\d+)\s*(?:символ(?:ів|и)?|знаків?)",
    ]
    q = question.lower()
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def random_text(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


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


def get_model_device() -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return next(model.parameters()).device


def get_image_patch_size() -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if isinstance(patch_size, int):
        return patch_size
    return 16


def build_image_message(image_path: str) -> dict[str, Any]:
    image_message: dict[str, Any] = {
        "type": "image",
        "image": Path(image_path).resolve().as_uri(),
    }
    if IMAGE_MIN_PIXELS is not None:
        image_message["min_pixels"] = IMAGE_MIN_PIXELS
    if IMAGE_MAX_PIXELS is not None:
        image_message["max_pixels"] = IMAGE_MAX_PIXELS
    return image_message


def build_generation_config() -> Any:
    generation_config = deepcopy(model.generation_config)
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.use_cache = True
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    if pad_token_id is not None:
        generation_config.pad_token_id = pad_token_id
    if eos_token_id is not None:
        generation_config.eos_token_id = eos_token_id

    return generation_config


def run_text_model(messages: list[dict[str, Any]]) -> str:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=text,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(get_model_device())

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            generation_config=build_generation_config(),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(output_text[0]).strip()


def run_vision_model(image_path: str, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_TEXT_VISION}],
        },
        {
            "role": "user",
            "content": [
                build_image_message(image_path),
                {"type": "text", "text": user_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(
        messages,
        image_patch_size=get_image_patch_size(),
    )
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        do_resize=False,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(get_model_device())

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            generation_config=build_generation_config(),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(output_text[0]).strip()


def build_parse_prompt(mode: str, question: str, requested_input_length: int | None = None) -> str:
    if mode in {"yes_no", "point"}:
        return (
            "Return exactly one JSON object:\n"
            '{"status":"ok","target_description":"...","comment":"short"}\n'
            "Fields:\n"
            "- status: ok or ambiguous\n"
            "- target_description: a short strict internal description for screenshot grounding\n"
            "- comment: short reason\n"
            "Rules:\n"
            "- preserve only attributes that are explicit or clearly implied\n"
            "- preserve visible text exactly as written\n"
            "- do not invent missing color, type, state, label, or position\n"
            "- if the user says there/туди/сюди without enough context, use ambiguous\n"
            "- target_description should be one short phrase, not a full sentence\n"
            f"Mode: {mode}\n"
            f"User command: {question}"
        )

    if mode == "input":
        length_hint = ""
        if requested_input_length is not None:
            length_hint = f"- if random text is requested and length is unclear, use random_length={requested_input_length}\\n"

        return (
            "Return exactly one JSON object:\n"
            '{"status":"ok","target_description":"...","input_text":"","generate_random_text":false,'
            '"random_length":null,"comment":"short"}\n'
            "Fields:\n"
            "- status: ok or ambiguous\n"
            "- target_description: strict description of the editable input field\n"
            "- input_text: literal text to type, or empty string if not explicitly provided\n"
            "- generate_random_text: true only if the user explicitly asks for random text\n"
            "- random_length: integer length for random text, or null\n"
            "- comment: short reason\n"
            "Rules:\n"
            "- preserve visible labels, placeholders, or exact text exactly as written\n"
            "- do not use a label as the target itself; the target is the editable field\n"
            "- if the command does not identify a unique target field, use ambiguous\n"
            "- if the command asks for random text, keep input_text empty and set generate_random_text=true\n"
            "- if the command provides literal text to enter, copy it exactly into input_text\n"
            f"{length_hint}"
            f"User command: {question}"
        )

    if mode == "drag":
        return (
            "Return exactly one JSON object:\n"
            '{"status":"ok","source_description":"...","destination_description":"...","comment":"short"}\n'
            "Fields:\n"
            "- status: ok or ambiguous\n"
            "- source_description: strict description of what should be dragged\n"
            "- destination_description: strict description of where it should be dropped\n"
            "- comment: short reason\n"
            "Rules:\n"
            "- both source_description and destination_description must be present for status=ok\n"
            "- preserve explicit text exactly as written\n"
            "- do not invent missing source or destination details\n"
            "- if the command does not identify both endpoints clearly enough, use ambiguous\n"
            f"User command: {question}"
        )

    raise ValueError("Unsupported mode")


def build_exists_prompt(target_description: str) -> str:
    return (
        "Return exactly one JSON object:\n"
        '{"status":"found","comment":"short"}\n'
        "Fields:\n"
        "- status: found, not_found, or ambiguous\n"
        "- comment: short reason\n"
        "Rules:\n"
        "- use the internal task as the source of truth\n"
        "- the target must satisfy all attributes in the description\n"
        "- if any required attribute is missing or mismatched, return not_found\n"
        "- if several candidates still match, return ambiguous\n"
        f"Internal task: {target_description}"
    )


def build_point_prompt(target_description: str) -> str:
    return (
        "Return exactly one JSON object:\n"
        '{"status":"found","x":500,"y":500,"comment":"short"}\n'
        "Fields:\n"
        f"- status: found, not_found, or ambiguous\n"
        f"- x: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        f"- y: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        "- comment: short reason\n"
        "Rules:\n"
        "- return normalized coordinates across the full image\n"
        "- return the center of the target element itself\n"
        "- for buttons, use the center of the clickable button rectangle\n"
        "- for input fields, use the center of the editable text box\n"
        "- do not return a nearby label, icon, text, or container\n"
        "- if any required attribute is missing or mismatched, return not_found with nulls\n"
        "- if several candidates still match, return ambiguous with nulls\n"
        "- never return coordinates for a merely similar element\n"
        f"Internal task: {target_description}"
    )


def build_drag_prompt(source_description: str, destination_description: str) -> str:
    return (
        "Return exactly one JSON object:\n"
        '{"status":"found","x":500,"y":500,"x2":800,"y2":500,"comment":"short"}\n'
        "Fields:\n"
        f"- status: found, not_found, or ambiguous\n"
        f"- x, y, x2, y2: integers from 0 to {NORMALIZED_COORD_MAX}, or null\n"
        "- comment: short reason\n"
        "Rules:\n"
        "- x,y is the center of the draggable source element\n"
        "- x2,y2 is the center of the destination drop point or destination element\n"
        "- both endpoints must match exactly\n"
        "- if either endpoint is missing, return not_found with nulls\n"
        "- if either endpoint is ambiguous, return ambiguous with nulls\n"
        "- never use a similar element as a fallback\n"
        f"Source task: {source_description}\n"
        f"Destination task: {destination_description}"
    )


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
            "comment": limit_reason_words(comment),
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
            "comment": limit_reason_words(data["comment"]),
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
            "comment": limit_reason_words(data["comment"]),
        }

    raise ValueError("Unsupported mode")


def validate_exists_output(data: dict[str, Any]) -> dict[str, Any]:
    if set(data.keys()) != {"status", "comment"}:
        raise ValueError("unexpected keys for exists output")
    if data["status"] not in {"found", "not_found", "ambiguous"}:
        raise ValueError("exists status must be found/not_found/ambiguous")
    if not isinstance(data["comment"], str):
        raise ValueError("comment must be string")
    return {
        "status": data["status"],
        "comment": limit_reason_words(data["comment"]),
    }


def _validate_normalized_int(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be integer or null")
    if not (0 <= value <= NORMALIZED_COORD_MAX):
        raise ValueError(f"{field_name} must be in [0, {NORMALIZED_COORD_MAX}]")


def validate_point_output(data: dict[str, Any]) -> dict[str, Any]:
    if set(data.keys()) != {"status", "x", "y", "comment"}:
        raise ValueError("unexpected keys for point output")
    if data["status"] not in {"found", "not_found", "ambiguous"}:
        raise ValueError("point status must be found/not_found/ambiguous")
    if not isinstance(data["comment"], str):
        raise ValueError("comment must be string")
    _validate_normalized_int(data["x"], "x")
    _validate_normalized_int(data["y"], "y")
    if data["status"] != "found" and (data["x"] is not None or data["y"] is not None):
        raise ValueError("non-found point output must use null coordinates")
    if data["status"] == "found" and (data["x"] is None or data["y"] is None):
        raise ValueError("found point output must include coordinates")
    return {
        "status": data["status"],
        "x": data["x"],
        "y": data["y"],
        "comment": limit_reason_words(data["comment"]),
    }


def validate_drag_output(data: dict[str, Any]) -> dict[str, Any]:
    if set(data.keys()) != {"status", "x", "y", "x2", "y2", "comment"}:
        raise ValueError("unexpected keys for drag output")
    if data["status"] not in {"found", "not_found", "ambiguous"}:
        raise ValueError("drag status must be found/not_found/ambiguous")
    if not isinstance(data["comment"], str):
        raise ValueError("comment must be string")
    for field_name in ("x", "y", "x2", "y2"):
        _validate_normalized_int(data[field_name], field_name)
    if data["status"] != "found" and any(data[k] is not None for k in ("x", "y", "x2", "y2")):
        raise ValueError("non-found drag output must use null coordinates")
    if data["status"] == "found" and any(data[k] is None for k in ("x", "y", "x2", "y2")):
        raise ValueError("found drag output must include all coordinates")
    return {
        "status": data["status"],
        "x": data["x"],
        "y": data["y"],
        "x2": data["x2"],
        "y2": data["y2"],
        "comment": limit_reason_words(data["comment"]),
    }


def call_parser(question: str, mode: str, max_retries: int = 3) -> dict[str, Any]:
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
            data = extract_json(raw_text)
            return validate_parser_output(data, mode)
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)

    raise RuntimeError(
        f"Parser failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )


def call_exists(image_path: str, target_description: str, max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for _attempt in range(1, max_retries + 1):
        prompt = build_exists_prompt(target_description)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            return validate_exists_output(data)
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)

    raise RuntimeError(
        f"Exists check failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )


def call_point(image_path: str, target_description: str, max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for _attempt in range(1, max_retries + 1):
        prompt = build_point_prompt(target_description)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            return validate_point_output(data)
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)

    raise RuntimeError(
        f"Point localization failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )


def call_drag(image_path: str, source_description: str, destination_description: str, max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for _attempt in range(1, max_retries + 1):
        prompt = build_drag_prompt(source_description, destination_description)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            return validate_drag_output(data)
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)

    raise RuntimeError(
        f"Drag localization failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"Last raw output: {last_raw_text!r}"
    )


def normalized_to_pixels(x: int, y: int, image_path: str) -> dict[str, int]:
    width, height = Image.open(image_path).size
    pixel_x = round(x * (width - 1) / NORMALIZED_COORD_MAX)
    pixel_y = round(y * (height - 1) / NORMALIZED_COORD_MAX)
    return {"x": pixel_x, "y": pixel_y}


def normalized_drag_to_pixels(x: int, y: int, x2: int, y2: int, image_path: str) -> dict[str, int]:
    width, height = Image.open(image_path).size
    return {
        "x": round(x * (width - 1) / NORMALIZED_COORD_MAX),
        "y": round(y * (height - 1) / NORMALIZED_COORD_MAX),
        "x2": round(x2 * (width - 1) / NORMALIZED_COORD_MAX),
        "y2": round(y2 * (height - 1) / NORMALIZED_COORD_MAX),
    }


def yes_no_result(answer: bool, comment: str) -> dict[str, Any]:
    return {
        "mode": "yes_no",
        "answer": answer,
        "comment": limit_reason_words(comment),
    }


def point_result(mode: str, answer: dict[str, Any], comment: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "answer": answer,
        "comment": clean_single_line_text(comment),
    }


def null_xy() -> dict[str, None]:
    return {"x": None, "y": None}


def null_drag() -> dict[str, None]:
    return {"x": None, "y": None, "x2": None, "y2": None}


def resolve_input_text(parsed: dict[str, Any], fallback_question: str) -> str:
    if parsed["generate_random_text"]:
        length = parsed["random_length"]
        if length is None:
            length = extract_requested_input_length(fallback_question) or DEFAULT_RANDOM_TEXT_LENGTH
        return random_text(length)
    return clean_single_line_text(parsed["input_text"])


def ask_image_json(
    image_path: str,
    question: str,
    mode: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    image_path = str(Path(image_path).expanduser().resolve())
    mode = normalize_mode(mode)

    parsed = call_parser(question, mode, max_retries=max_retries)

    if mode == "yes_no":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            return yes_no_result(False, parsed["comment"] or "ambiguous command")

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        return yes_no_result(exists["status"] == "found", exists["comment"])

    if mode == "point":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            return point_result("point", null_xy(), parsed["comment"] or "ambiguous command")

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        if exists["status"] != "found":
            return point_result("point", null_xy(), exists["comment"])

        point = call_point(image_path, parsed["target_description"], max_retries=max_retries)
        if point["status"] != "found" or point["x"] is None or point["y"] is None:
            return point_result("point", null_xy(), point["comment"])

        return point_result(
            "point",
            normalized_to_pixels(point["x"], point["y"], image_path),
            point["comment"],
        )

    if mode == "input":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            return point_result("input", null_xy(), parsed["comment"] or "ambiguous command")

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        if exists["status"] != "found":
            return point_result("input", null_xy(), exists["comment"])

        point = call_point(image_path, parsed["target_description"], max_retries=max_retries)
        if point["status"] != "found" or point["x"] is None or point["y"] is None:
            return point_result("input", null_xy(), point["comment"])

        input_text = resolve_input_text(parsed, question)
        return point_result(
            "input",
            normalized_to_pixels(point["x"], point["y"], image_path),
            input_text,
        )

    if mode == "drag":
        if (
            parsed["status"] != "ok"
            or not parsed["source_description"]
            or not parsed["destination_description"]
        ):
            return point_result("drag", null_drag(), parsed["comment"] or "ambiguous command")

        source_exists = call_exists(image_path, parsed["source_description"], max_retries=max_retries)
        if source_exists["status"] != "found":
            return point_result("drag", null_drag(), f"source {source_exists['comment']}")

        destination_exists = call_exists(image_path, parsed["destination_description"], max_retries=max_retries)
        if destination_exists["status"] != "found":
            return point_result("drag", null_drag(), f"destination {destination_exists['comment']}")

        drag = call_drag(
            image_path,
            parsed["source_description"],
            parsed["destination_description"],
            max_retries=max_retries,
        )
        if drag["status"] != "found":
            return point_result("drag", null_drag(), drag["comment"])

        return point_result(
            "drag",
            normalized_drag_to_pixels(
                drag["x"],
                drag["y"],
                drag["x2"],
                drag["y2"],
                image_path,
            ),
            drag["comment"],
        )

    raise ValueError("Unsupported mode")


if __name__ == "__main__":
    img = "/home/total/Pictures/Screenshots/scr4.png"

    examples = [
        ("yes_no", 'Logo with smiley face should be in top left corner of image'),
        ("yes_no", 'Logo with sad face should be in top left corner of image'),
        ("yes_no", 'Logo with text "ROZETKA" should be in top left corner of image'),
        ("yes_no", 'Logo with text "fjgdfkj" should be in top left corner of image'),
        ("yes_no", 'Is there a green button with text "Знайти" visible on the screenshot?'),
        ("yes_no", 'Is there a red button with text "Знайти" visible on the screenshot?'),
        ("point", 'Point to green button with text "Знайти"'),
        ("point", 'Point to rectangular button with text "Знайти"'),
        ("point", 'Point to red button with text "Знайти"'),
        ("point", 'Point to round button with text "Знайти"'),
        ("point", "Return the center point of the Upload button",),
        ("input", 'input some random text with length of 12 symbols in input box below Бренд label'),
        ("drag", 'Drag the left handle of the price slider to the middle of the slider track'),
        ("drag", 'Drag the left handle of the "ціна" slider to the middle of the slider track')
    ]

    i = 0
    for mode_name, question_text in examples:
        result = ask_image_json(img, question_text, mode_name)
        print(json.dumps(result, ensure_ascii=False))
        i+=1
        if result['mode'] in ("input", "point", "drag"):
            draw(result["answer"], img, f"/home/total/ai/qwen/qwen6_q{i}.png")
