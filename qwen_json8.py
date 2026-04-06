import json
import os
import platform
import random
import re
import string
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from qwen_logging import log_event, log_resource_snapshot
from qwen_parser7 import (
    clean_single_line_text,
    extract_json,
    extract_requested_input_length,
)
from qwen_prompts import (
    SYSTEM_TEXT_GROUND_V2,
)
from qwen_schemas import (
    validate_drag_output as validate_drag_output_schema,
    validate_exists_output as validate_exists_output_schema,
    validate_multi_value_output as validate_multi_value_output_schema,
    validate_point_output as validate_point_output_schema,
    validate_value_output as validate_value_output_schema,
)
from raw_answer_point import draw

REQUESTED_BACKEND = os.environ.get("QWEN_BACKEND", "auto").strip().lower()
if REQUESTED_BACKEND in {"auto", "rocm"}:
    # This flag must be present before ROCm attention kernels are selected.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")
OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", "test_images/output"))

IMAGE_MIN_PIXELS = 256 * 256
IMAGE_MAX_PIXELS = 1200 * 1200

GROUND_MAX_NEW_TOKENS_BY_MODE = {
    "yes_no": 96,
    "point": 96,
    "input": 96,
    "drag": 128,
    "value": 160,
    "multi_value": 384,
}
NORMALIZED_COORD_MAX = 1000
DEFAULT_RANDOM_TEXT_LENGTH = 12
BACKEND_ALIASES = {"auto", "cuda", "rocm"}
IS_WINDOWS = platform.system() == "Windows"


def detect_runtime_backend() -> str:
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "rocm"
        return "cuda"
    return "unavailable"


def resolve_dtype(backend: str) -> torch.dtype:
    if backend == "rocm" and IS_WINDOWS:
        # Windows ROCm currently exposes FP16 reliably for inference workloads.
        return torch.float16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


if REQUESTED_BACKEND not in BACKEND_ALIASES:
    raise ValueError("QWEN_BACKEND must be one of: auto, cuda, rocm")

RUNTIME_BACKEND = detect_runtime_backend()
if REQUESTED_BACKEND == "auto":
    if RUNTIME_BACKEND == "unavailable":
        raise RuntimeError("No supported GPU backend detected. Available backends are: cuda, rocm.")
    ACTIVE_BACKEND = RUNTIME_BACKEND
elif RUNTIME_BACKEND != REQUESTED_BACKEND:
    raise RuntimeError(
        f"Requested backend '{REQUESTED_BACKEND}' is not available. "
        f"Detected backend: '{RUNTIME_BACKEND}'."
    )
else:
    ACTIVE_BACKEND = REQUESTED_BACKEND

DEVICE_MAP = "auto"
DTYPE = resolve_dtype(ACTIVE_BACKEND)
ATTN_IMPLEMENTATION = resolve_attn_implementation()

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
        "rocm_aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
        "detected_backend": RUNTIME_BACKEND,
        "active_backend": ACTIVE_BACKEND,
        "device_map": DEVICE_MAP,
        "dtype": str(DTYPE),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "model_path": MODEL_PATH,
    },
)
log_resource_snapshot("runtime.resources", torch_module=torch, model=model)


def random_text(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


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
        "value": "value",
        "multi_value": "multi_value",
    }
    if value not in aliases:
        raise ValueError("mode must be one of: yes_no, y_n, point, input, drag, value, multi_value")
    return aliases[value]


def get_model_device() -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return next(model.parameters()).device


def synchronize_device() -> None:
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def get_image_patch_size() -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if isinstance(patch_size, int):
        return patch_size
    return 16


def resolve_image_reference(image_path: str) -> str:
    image_ref = image_path.strip()
    if image_ref.startswith(("http://", "https://", "data:image", "file://")):
        return image_ref
    return str(Path(image_ref).expanduser().resolve())


def build_image_message(image_path: str) -> dict[str, Any]:
    image_message: dict[str, Any] = {
        "type": "image",
        # qwen_vl_utils strips "file://" manually and breaks Windows paths like
        # "file:///D:/..." into "/D:/...". Use a native absolute path instead.
        "image": resolve_image_reference(image_path),
    }
    if IMAGE_MIN_PIXELS is not None:
        image_message["min_pixels"] = IMAGE_MIN_PIXELS
    if IMAGE_MAX_PIXELS is not None:
        image_message["max_pixels"] = IMAGE_MAX_PIXELS
    return image_message


def resolve_ground_max_new_tokens(mode: str) -> int:
    try:
        return GROUND_MAX_NEW_TOKENS_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def build_generation_config(max_new_tokens: int) -> Any:
    generation_config = deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
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


def get_output_dir() -> Path:
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_result_record(output_path: Path, payload: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_vision_model(image_path: str, user_text: str, mode: str) -> str:
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
            "content": [{"type": "text", "text": SYSTEM_TEXT_GROUND_V2}],
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

    synchronize_device()
    started_at = perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            generation_config=build_generation_config(resolve_ground_max_new_tokens(mode)),
        )
    synchronize_device()
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    input_token_count = int(inputs.input_ids.shape[-1])
    output_token_count = sum(len(out_ids) for out_ids in generated_ids_trimmed)
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    result = str(output_text[0]).strip()
    log_resource_snapshot(
        "vision_model.perf",
        torch_module=torch,
        model=model,
        extra={
            "elapsed_ms": elapsed_ms,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "mode": mode,
            "image_count": len(images),
            "video_count": len(videos),
        },
    )
    log_event("vision_model.output", payload=result)
    return result


def build_direct_ground_prompt(mode: str, question: str) -> str:
    prompt_header = (
        "You receive one screenshot and one free-form user request.\n"
        "The required JSON schema and task rules are defined below.\n"
        "Interpret the user's wording only as much as needed to complete that task on the screenshot.\n"
        "Use only what is visible in the screenshot.\n"
        "Do not invent missing attributes, hidden values, or off-screen content.\n"
        "Return exactly one valid JSON object and nothing else.\n"
        f"User request: {question}\n"
    )

    if mode == "yes_no":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- comment: string; brief reason why the target was found, not found, or ambiguous\n"
            "Rules:\n"
            "- decide whether the screenshot contains the target implied by the user request\n"
            "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
            "- if there is no exact visible match, return not_found\n"
            "- if several candidates still match, return ambiguous\n"
            "- do not relax the request to a merely similar element"
        )

    if mode == "point":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            f"- x: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
            f"- y: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
            "- comment: string; brief reason why the coordinates were returned or why they are null\n"
            "Rules:\n"
            "- locate the single target element implied by the user request\n"
            "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
            "- return normalized coordinates across the full image\n"
            "- return the center of the target element itself\n"
            "- do not return a nearby label, icon, text, or container\n"
            "- if there is no exact visible match, return not_found with nulls\n"
            "- if several candidates still match, return ambiguous with nulls\n"
            "- do not return coordinates for a merely similar element"
        )

    if mode == "input":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            f"- x: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
            f"- y: integer from 0 to {NORMALIZED_COORD_MAX}, or null\n"
            "- comment: string; brief reason why the coordinates were returned or why they are null\n"
            "Rules:\n"
            "- identify the editable field the user wants to type into\n"
            "- the free-form request may contain both field description and text to type\n"
            "- use typing text only when it also helps identify the target field explicitly\n"
            "- return normalized coordinates across the full image\n"
            "- return the center of the editable input field itself\n"
            "- do not return a nearby label, icon, text, or container\n"
            "- if there is no exact visible match, return not_found with nulls\n"
            "- if several candidates still match, return ambiguous with nulls\n"
            "- do not treat the requested text value itself as proof that the field is present"
        )

    if mode == "drag":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            f"- x, y, x2, y2: integers from 0 to {NORMALIZED_COORD_MAX}, or null\n"
            "- comment: string; brief reason why the drag coordinates were returned or why they are null\n"
            "Rules:\n"
            "- infer the draggable source and destination directly from the free-form user request\n"
            "- x,y is the center of the draggable source element\n"
            "- x2,y2 is the center of the destination drop point or destination element\n"
            "- preserve explicit text, color, position, relation, and state constraints from the user request\n"
            "- both endpoints must match the request exactly\n"
            "- if either endpoint is missing, return not_found with nulls\n"
            "- if either endpoint is ambiguous, return ambiguous with nulls\n"
            "- never use a merely similar source or destination as a fallback\n"
            "- do not substitute a generic center point when the destination is underspecified"
        )

    if mode == "value":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- answer: string; the exact visible value, or empty string if not_found or ambiguous\n"
            "- comment: string; brief reason why the value was returned or why it is empty\n"
            "Rules:\n"
            "- identify from the free-form request which single visible value should be read\n"
            "- return exactly one visible value from the requested element or region\n"
            "- preserve visible text exactly as written\n"
            "- if the required target is missing or mismatched, return not_found with answer=''\n"
            "- if several candidates still match, return ambiguous with answer=''\n"
            "- do not guess hidden, cropped, or inferred values\n"
            "- do not return labels or surrounding text unless they are the value itself"
        )

    if mode == "multi_value":
        return (
            f"{prompt_header}"
            "Required keys:\n"
            "- status: found, not_found, or ambiguous\n"
            "- answer: array of strings; the visible items in order, or [] if not_found or ambiguous\n"
            "- comment: string; brief reason why the list was returned or why it is empty\n"
            "Rules:\n"
            "- identify from the free-form request which list, group, or collection should be read\n"
            "- return only the visible items that belong to the requested list or group\n"
            "- preserve visible text exactly as written\n"
            "- return items in visual order from top to bottom, or left to right when appropriate\n"
            "- if the required target is missing or mismatched, return not_found with answer=[]\n"
            "- if several candidate lists or groups still match, return ambiguous with answer=[]\n"
            "- do not guess hidden, cropped, or inferred items\n"
            "- do not merge items from different groups or sections"
        )

    raise ValueError(f"Unsupported mode: {mode}")

def validate_exists_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_exists_output_schema(data)
    return {
        "status": validated["status"],
        "comment": validated["comment"],
    }


def validate_point_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_point_output_schema(data, NORMALIZED_COORD_MAX)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "comment": validated["comment"],
    }


def validate_value_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_value_output_schema(data)
    return {
        "status": validated["status"],
        "answer": validated["answer"],
        "comment": validated["comment"],
    }


def validate_multi_value_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_multi_value_output_schema(data)
    return {
        "status": validated["status"],
        "answer": validated["answer"],
        "comment": validated["comment"],
    }


def validate_drag_output(data: dict[str, Any]) -> dict[str, Any]:
    validated = validate_drag_output_schema(data, NORMALIZED_COORD_MAX)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "x2": validated["x2"],
        "y2": validated["y2"],
        "comment": validated["comment"],
    }

def ground_action(image_path: str, question: str, mode: str) -> dict[str, Any]:
    prompt = build_direct_ground_prompt(mode, question)
    log_event(
        "ground.request",
        payload={
            "image_path": image_path,
            "question": question,
            "prompt": prompt,
        },
    )

    raw_text = run_vision_model(image_path, prompt, mode)
    log_event("ground.raw_output", payload=raw_text)
    data = extract_json(raw_text)
    if mode == "yes_no":
        validated = validate_exists_output(data)
    elif mode in {"point", "input"}:
        validated = validate_point_output(data)
    elif mode == "value":
        validated = validate_value_output(data)
    elif mode == "multi_value":
        validated = validate_multi_value_output(data)
    elif mode == "drag":
        validated = validate_drag_output(data)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    log_event("ground.result", payload=validated)
    return validated


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


def null_xy() -> dict[str, None]:
    return {"x": None, "y": None}


def null_drag() -> dict[str, None]:
    return {"x": None, "y": None, "x2": None, "y2": None}


def action_result(mode: str, status: str, answer: Any, comment: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": status,
        "answer": answer,
        "comment": comment,
    }


def resolve_input_text(question: str) -> str:
    if re.search(r"\brandom\b", question, flags=re.IGNORECASE):
        length = extract_requested_input_length(question) or DEFAULT_RANDOM_TEXT_LENGTH
        generated = random_text(length)
        log_event(
            "input.generated_text",
            payload={"length": length, "text": generated},
        )
        return generated

    quoted_match = re.search(r'"([^"]+)"', question)
    if quoted_match:
        resolved = clean_single_line_text(quoted_match.group(1))
        log_event("input.literal_text", payload=resolved)
        return resolved

    quoted_match = re.search(r"'([^']+)'", question)
    if quoted_match:
        resolved = clean_single_line_text(quoted_match.group(1))
        log_event("input.literal_text", payload=resolved)
        return resolved

    verb_match = re.search(
        r"^\s*(?:input|type|enter|fill(?:\s+in)?)\s+(.+?)(?:\s+(?:in|into|inside)\b.*)?$",
        question,
        flags=re.IGNORECASE,
    )
    if verb_match:
        resolved = clean_single_line_text(verb_match.group(1))
        log_event("input.literal_text", payload=resolved)
        return resolved

    resolved = ""
    log_event("input.literal_text", payload=resolved)
    return resolved


def ask_image_json(image_path: str, question: str, mode: str) -> dict[str, Any]:
    image_path = str(Path(image_path).expanduser().resolve())
    mode = normalize_mode(mode)
    log_event(
        "ask.start",
        payload={
            "image_path": image_path,
            "question": question,
            "mode": mode,
        },
    )

    grounded = ground_action(image_path, question, mode)

    if mode == "yes_no":
        answer = grounded["status"] == "found"
        if grounded["status"] == "ambiguous":
            answer = None
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "point":
        answer = null_xy()
        if grounded["status"] == "found" and grounded["x"] is not None and grounded["y"] is not None:
            answer = normalized_to_pixels(grounded["x"], grounded["y"], image_path)
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "input":
        text_to_type = resolve_input_text(question)
        answer: dict[str, Any] = {"x": None, "y": None, "text": text_to_type}
        if grounded["status"] == "found" and grounded["x"] is not None and grounded["y"] is not None:
            answer.update(normalized_to_pixels(grounded["x"], grounded["y"], image_path))
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "drag":
        answer = null_drag()
        if grounded["status"] == "found":
            answer = normalized_drag_to_pixels(
                grounded["x"],
                grounded["y"],
                grounded["x2"],
                grounded["y2"],
                image_path,
            )
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "value":
        answer = grounded["answer"] if grounded["status"] == "found" else ""
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    if mode == "multi_value":
        answer = grounded["answer"] if grounded["status"] == "found" else []
        result = action_result(mode, grounded["status"], answer, grounded["comment"])
        log_event("ask.result", payload=result)
        return result

    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    img = "test_images/input/rozetka_50.png"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_dir()
    results_path = output_dir / f"qwen7_results_{run_timestamp}.jsonl"

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
        ("value", "how many items shown in shopping cart in top right corner?"),
        ("multi_value", "What items displayed in Popular brands list?"),
    ]

    for i, (mode_name, question_text) in enumerate(examples, start=1):
        result = ask_image_json(img, question_text, mode_name)
        append_result_record(
            results_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_path": str(Path(img).resolve()),
                "mode": mode_name,
                "question": question_text,
                "result": result,
            },
        )
        print(json.dumps(result, ensure_ascii=False))
        if result["mode"] in {"point", "input", "drag"}:
            draw(result["answer"], img, output_dir / f"qwen7_{run_timestamp}_q{i}.png")
    print(json.dumps({"results_file": str(results_path)}, ensure_ascii=False))
