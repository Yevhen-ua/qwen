import json
import os
import random
import string
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_logging import log_event, log_retry
from qwen_parser import call_parser, clean_single_line_text, extract_json, extract_requested_input_length
from qwen_prompts import (
    SYSTEM_TEXT_VISION,
    build_drag_prompt,
    build_exists_prompt,
    build_point_prompt,
)
from qwen_schemas import (
    validate_drag_output as validate_drag_output_schema,
    validate_exists_output as validate_exists_output_schema,
    validate_point_output as validate_point_output_schema,
)
from raw_answer_point import draw

REQUESTED_BACKEND = os.environ.get("QWEN_BACKEND", "auto").strip().lower()
if REQUESTED_BACKEND == "rocm":
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")

# Image resizing control for Qwen3-VL.
IMAGE_MIN_PIXELS = 256 * 256
IMAGE_MAX_PIXELS = 1200 * 1200

MAX_NEW_TOKENS = 120
NORMALIZED_COORD_MAX = 1000
DEFAULT_RANDOM_TEXT_LENGTH = 12
BACKEND_ALIASES = {"auto", "cpu", "cuda", "rocm"}


def detect_runtime_backend() -> str:
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "rocm"
        if getattr(torch.version, "cuda", None):
            return "cuda"
        return "gpu"
    return "cpu"


def resolve_dtype(backend: str) -> torch.dtype:
    if backend == "cpu":
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_attn_implementation(backend: str) -> str:
    if backend == "cpu":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


if REQUESTED_BACKEND not in BACKEND_ALIASES:
    raise ValueError("QWEN_BACKEND must be one of: auto, cpu, cuda, rocm")

RUNTIME_BACKEND = detect_runtime_backend()
if REQUESTED_BACKEND == "auto":
    ACTIVE_BACKEND = RUNTIME_BACKEND
elif REQUESTED_BACKEND == "cpu":
    ACTIVE_BACKEND = "cpu"
elif RUNTIME_BACKEND != REQUESTED_BACKEND:
    raise RuntimeError(
        f"Requested backend '{REQUESTED_BACKEND}' is not available. "
        f"Detected backend: '{RUNTIME_BACKEND}'."
    )
else:
    ACTIVE_BACKEND = REQUESTED_BACKEND

DEVICE_MAP = "cpu" if ACTIVE_BACKEND == "cpu" else "auto"
DTYPE = resolve_dtype(ACTIVE_BACKEND)
ATTN_IMPLEMENTATION = resolve_attn_implementation(ACTIVE_BACKEND)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    device_map=DEVICE_MAP,
    attn_implementation=ATTN_IMPLEMENTATION,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
log_event(
    "runtime.config",
    payload={
        "requested_backend": REQUESTED_BACKEND,
        "detected_backend": RUNTIME_BACKEND,
        "active_backend": ACTIVE_BACKEND,
        "device_map": DEVICE_MAP,
        "dtype": str(DTYPE),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "model_path": MODEL_PATH,
    },
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


def random_text(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


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
    log_event("text_model.messages", payload=messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    log_event("text_model.prompt", payload=text)
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
    result = str(output_text[0]).strip()
    log_event("text_model.output", payload=result)
    return result


def run_vision_model(image_path: str, user_text: str) -> str:
    log_event(
        "vision_model.request",
        payload={
            "image_path": image_path,
            "prompt": user_text,
        },
    )
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
    result = str(output_text[0]).strip()
    log_event("vision_model.output", payload=result)
    return result


def validate_exists_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_exists_output_schema(data)
    log_event("exists.validated", payload=validated)
    return {
        "status": validated["status"],
        "comment": validated["comment"],
    }


def validate_point_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_point_output_schema(data, NORMALIZED_COORD_MAX)
    log_event("point.validated", payload=validated)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "comment": validated["comment"],
    }


def validate_drag_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_drag_output_schema(data, NORMALIZED_COORD_MAX)
    log_event("drag.validated", payload=validated)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "x2": validated["x2"],
        "y2": validated["y2"],
        "comment": validated["comment"],
    }


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
        log_event(
            "exists.request",
            f"attempt {_attempt}/{max_retries}",
            {
                "image_path": image_path,
                "target_description": target_description,
                "prompt": prompt,
            },
        )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            log_event("exists.json", payload=data)
            validated = validate_exists_output(data)
            log_event("exists.result", payload=validated)
            return validated
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("exists", _attempt, max_retries, exc, last_raw_text)

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
        prompt = build_point_prompt(target_description, NORMALIZED_COORD_MAX)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )
        log_event(
            "point.request",
            f"attempt {_attempt}/{max_retries}",
            {
                "image_path": image_path,
                "target_description": target_description,
                "prompt": prompt,
            },
        )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            log_event("point.json", payload=data)
            validated = validate_point_output(data)
            log_event("point.result", payload=validated)
            return validated
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("point", _attempt, max_retries, exc, last_raw_text)

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
        prompt = build_drag_prompt(
            source_description,
            destination_description,
            NORMALIZED_COORD_MAX,
        )
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )
        log_event(
            "drag.request",
            f"attempt {_attempt}/{max_retries}",
            {
                "image_path": image_path,
                "source_description": source_description,
                "destination_description": destination_description,
                "prompt": prompt,
            },
        )
        try:
            raw_text = run_vision_model(image_path, prompt)
            last_raw_text = raw_text
            data = extract_json(raw_text)
            log_event("drag.json", payload=data)
            validated = validate_drag_output(data)
            log_event("drag.result", payload=validated)
            return validated
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("drag", _attempt, max_retries, exc, last_raw_text)

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
        "comment": comment,
    }


def point_result(mode: str, answer: dict[str, Any], comment: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "answer": answer,
        "comment": comment,
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
        generated = random_text(length)
        log_event(
            "input.generated_text",
            payload={"length": length, "text": generated},
        )
        return generated
    resolved = clean_single_line_text(parsed["input_text"])
    log_event("input.literal_text", payload=resolved)
    return resolved


def ask_image_json(
    image_path: str,
    question: str,
    mode: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    image_path = str(Path(image_path).expanduser().resolve())
    mode = normalize_mode(mode)
    log_event(
        "ask.start",
        payload={
            "image_path": image_path,
            "question": question,
            "mode": mode,
            "max_retries": max_retries,
        },
    )

    parsed = call_parser(question, mode, run_text_model, max_retries=max_retries)
    log_event("ask.parsed", payload=parsed)

    if mode == "yes_no":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            result = yes_no_result(False, parsed["comment"] or "ambiguous command")
            log_event("ask.result", payload=result)
            return result

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        result = yes_no_result(exists["status"] == "found", exists["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "point":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            result = point_result("point", null_xy(), parsed["comment"] or "ambiguous command")
            log_event("ask.result", payload=result)
            return result

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        if exists["status"] != "found":
            result = point_result("point", null_xy(), exists["comment"])
            log_event("ask.result", payload=result)
            return result

        point = call_point(image_path, parsed["target_description"], max_retries=max_retries)
        if point["status"] != "found" or point["x"] is None or point["y"] is None:
            result = point_result("point", null_xy(), point["comment"])
            log_event("ask.result", payload=result)
            return result

        result = point_result(
            "point",
            normalized_to_pixels(point["x"], point["y"], image_path),
            point["comment"],
        )
        log_event("ask.result", payload=result)
        return result

    if mode == "input":
        if parsed["status"] != "ok" or not parsed["target_description"]:
            result = point_result("input", null_xy(), parsed["comment"] or "ambiguous command")
            log_event("ask.result", payload=result)
            return result

        exists = call_exists(image_path, parsed["target_description"], max_retries=max_retries)
        if exists["status"] != "found":
            result = point_result("input", null_xy(), exists["comment"])
            log_event("ask.result", payload=result)
            return result

        point = call_point(image_path, parsed["target_description"], max_retries=max_retries)
        if point["status"] != "found" or point["x"] is None or point["y"] is None:
            result = point_result("input", null_xy(), point["comment"])
            log_event("ask.result", payload=result)
            return result

        input_text = resolve_input_text(parsed, question)
        result = point_result(
            "input",
            normalized_to_pixels(point["x"], point["y"], image_path),
            input_text,
        )
        log_event("ask.result", payload=result)
        return result

    if mode == "drag":
        if (
            parsed["status"] != "ok"
            or not parsed["source_description"]
            or not parsed["destination_description"]
        ):
            result = point_result("drag", null_drag(), parsed["comment"] or "ambiguous command")
            log_event("ask.result", payload=result)
            return result

        source_exists = call_exists(image_path, parsed["source_description"], max_retries=max_retries)
        if source_exists["status"] != "found":
            result = point_result("drag", null_drag(), f"source {source_exists['comment']}")
            log_event("ask.result", payload=result)
            return result

        destination_exists = call_exists(image_path, parsed["destination_description"], max_retries=max_retries)
        if destination_exists["status"] != "found":
            result = point_result("drag", null_drag(), f"destination {destination_exists['comment']}")
            log_event("ask.result", payload=result)
            return result

        drag = call_drag(
            image_path,
            parsed["source_description"],
            parsed["destination_description"],
            max_retries=max_retries,
        )
        if drag["status"] != "found":
            result = point_result("drag", null_drag(), drag["comment"])
            log_event("ask.result", payload=result)
            return result

        result = point_result(
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
        log_event("ask.result", payload=result)
        return result

    raise ValueError("Unsupported mode")


if __name__ == "__main__":
    img = "/home/total/Pictures/Screenshots/scr4.png"

    examples = [
        ("yes_no", 'Logo with smiley face should be in top left corner of image'),
        ("yes_no", 'Logo with sad face should be in top left corner of image'),
        ("yes_no", 'Logo with text "ROZETKA" should be in top left corner of image'),
        ("yes_no", 'Logo with text "fjgdfkj" should be in top left corner of image'),
        ("yes_no", 'Is there a green button with text "Find" visible on the screenshot?'),
        ("yes_no", 'Is there a red button with text "Find" visible on the screenshot?'),
        ("point", 'Point to green button with text "Find"'),
        ("point", 'Point to rectangular button with text "Find"'),
        ("point", 'Point to red button with text "Find"'),
        ("point", 'Point to round button with text "Find"'),
        ("point", "Return the center point of the Upload button",),
        ("input", 'input some random text with length of 12 symbols in input box below Brand label'),
        ("drag", 'Drag the left handle of the price slider to the middle of the slider track'),
        ("drag", 'Drag the left handle of the "price" slider to the middle of the slider track')
    ]

    i = 0
    for mode_name, question_text in examples:
        result = ask_image_json(img, question_text, mode_name)
        print(json.dumps(result, ensure_ascii=False))
        i+=1
        if result['mode'] in ("input", "point", "drag"):
            draw(result["answer"], img, f"/home/total/ai/qwen/qwen6_q{i}.png")
