import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
VLLM_CONFIG_PATH = Path("qwen_vllm_runtime_config.json")
with VLLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
    VLLM_RUNTIME_CONFIG = json.load(handle)

if VLLM_RUNTIME_CONFIG["cuda_visible_devices"] and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(VLLM_RUNTIME_CONFIG["cuda_visible_devices"])

import torch
from vllm import LLM, SamplingParams

from qwen_logging import log_event, log_resource_snapshot, log_retry
from qwen_parser7 import (
    clean_single_line_text,
    extract_json,
    extract_requested_input_length,
    interpret_command,
)
from qwen_prompts import SYSTEM_TEXT_GROUND_V2, build_ground_prompt_v2
from qwen_schemas import (
    validate_drag_output as validate_drag_output_schema,
    validate_exists_output as validate_exists_output_schema,
    validate_multi_value_output as validate_multi_value_output_schema,
    validate_point_output as validate_point_output_schema,
    validate_value_output as validate_value_output_schema,
)
from raw_answer_point import draw

MODEL_PATH = str(VLLM_RUNTIME_CONFIG["model_path"])
OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", "test_images/output"))

INTERPRET_MAX_NEW_TOKENS = int(os.environ.get("QWEN_INTERPRET_MAX_NEW_TOKENS", "160"))
GROUND_MAX_NEW_TOKENS_BY_MODE = {
    "yes_no": int(os.environ.get("QWEN_GROUND_YES_NO_MAX_NEW_TOKENS", "96")),
    "point": int(os.environ.get("QWEN_GROUND_POINT_MAX_NEW_TOKENS", "96")),
    "input": int(os.environ.get("QWEN_GROUND_INPUT_MAX_NEW_TOKENS", "96")),
    "drag": int(os.environ.get("QWEN_GROUND_DRAG_MAX_NEW_TOKENS", "128")),
    "value": int(os.environ.get("QWEN_GROUND_VALUE_MAX_NEW_TOKENS", "160")),
    "multi_value": int(os.environ.get("QWEN_GROUND_MULTI_VALUE_MAX_NEW_TOKENS", "384")),
}
NORMALIZED_COORD_MAX = 1000
DEFAULT_RANDOM_TEXT_LENGTH = 12

VLLM_TENSOR_PARALLEL_SIZE = int(VLLM_RUNTIME_CONFIG["tensor_parallel_size"])
VLLM_AUTO_DEFAULTS = VLLM_RUNTIME_CONFIG["auto_defaults"]
VLLM_DTYPE = str(VLLM_RUNTIME_CONFIG["dtype"])
VLLM_GPU_MEMORY_UTILIZATION = float(VLLM_RUNTIME_CONFIG["gpu_memory_utilization"])
VLLM_CPU_OFFLOAD_GB = float(VLLM_RUNTIME_CONFIG["cpu_offload_gb"])
VLLM_MAX_MODEL_LEN = int(VLLM_RUNTIME_CONFIG["max_model_len"])
VLLM_MAX_NUM_SEQS = int(VLLM_RUNTIME_CONFIG["max_num_seqs"])
VLLM_LIMIT_MM_IMAGES = int(VLLM_RUNTIME_CONFIG["limit_mm_per_prompt"]["image"])
VLLM_LIMIT_MM_VIDEOS = int(VLLM_RUNTIME_CONFIG["limit_mm_per_prompt"]["video"])
VLLM_MM_PROCESSOR_CACHE_GB = float(VLLM_RUNTIME_CONFIG["mm_processor_cache_gb"])
VLLM_MM_PROCESSOR_CACHE_TYPE = str(VLLM_RUNTIME_CONFIG["mm_processor_cache_type"] or "").strip()
VLLM_MM_ENCODER_TP_MODE = str(VLLM_RUNTIME_CONFIG["mm_encoder_tp_mode"] or "").strip()

llm_kwargs: dict[str, Any] = {
    "model": MODEL_PATH,
    "trust_remote_code": True,
    "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
    "dtype": VLLM_DTYPE,
    "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
    "cpu_offload_gb": VLLM_CPU_OFFLOAD_GB,
    "max_model_len": VLLM_MAX_MODEL_LEN,
    "max_num_seqs": VLLM_MAX_NUM_SEQS,
    "limit_mm_per_prompt": {
        "image": VLLM_LIMIT_MM_IMAGES,
        "video": VLLM_LIMIT_MM_VIDEOS,
    },
    "mm_processor_cache_gb": VLLM_MM_PROCESSOR_CACHE_GB,
}
if VLLM_MM_PROCESSOR_CACHE_TYPE:
    llm_kwargs["mm_processor_cache_type"] = VLLM_MM_PROCESSOR_CACHE_TYPE
if VLLM_MM_ENCODER_TP_MODE:
    llm_kwargs["mm_encoder_tp_mode"] = VLLM_MM_ENCODER_TP_MODE

_llm: LLM | None = None
_runtime_logged = False


def get_llm() -> LLM:
    global _llm, _runtime_logged
    if _llm is None:
        _llm = LLM(**llm_kwargs)

    if not _runtime_logged:
        log_event(
            "runtime.config",
            payload={
                "model_path": MODEL_PATH,
                "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
                "dtype": VLLM_DTYPE,
                "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
                "cpu_offload_gb": VLLM_CPU_OFFLOAD_GB,
                "max_model_len": VLLM_MAX_MODEL_LEN,
                "max_num_seqs": VLLM_MAX_NUM_SEQS,
                "limit_mm_per_prompt": {
                    "image": VLLM_LIMIT_MM_IMAGES,
                    "video": VLLM_LIMIT_MM_VIDEOS,
                },
                "mm_processor_cache_gb": VLLM_MM_PROCESSOR_CACHE_GB,
                "mm_processor_cache_type": VLLM_MM_PROCESSOR_CACHE_TYPE or None,
                "mm_encoder_tp_mode": VLLM_MM_ENCODER_TP_MODE or None,
                "cuda_visible_devices": VLLM_RUNTIME_CONFIG["cuda_visible_devices"],
                "config_path": str(VLLM_CONFIG_PATH.resolve()),
                "auto_defaults": VLLM_AUTO_DEFAULTS or None,
            },
        )
        log_resource_snapshot("runtime.resources", torch_module=torch)
        _runtime_logged = True

    return _llm


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


def resolve_ground_max_new_tokens(mode: str) -> int:
    try:
        return GROUND_MAX_NEW_TOKENS_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def get_output_dir() -> Path:
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_result_record(output_path: Path, payload: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )


def flatten_text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"Unsupported message content: {content!r}")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def request_output_text(request_output: Any) -> str:
    if not getattr(request_output, "outputs", None):
        raise RuntimeError("vLLM returned no outputs")
    return str(request_output.outputs[0].text).strip()


def request_output_token_count(request_output: Any) -> int | None:
    if not getattr(request_output, "outputs", None):
        return None
    token_ids = getattr(request_output.outputs[0], "token_ids", None)
    if token_ids is None:
        return None
    return len(token_ids)


def run_text_model(messages: list[dict[str, Any]]) -> str:
    llm = get_llm()
    log_event("text_model.messages", payload=messages)
    chat_messages = [
        {
            "role": str(message["role"]),
            "content": flatten_text_content(message),
        }
        for message in messages
    ]
    log_event("text_model.prompt", payload=chat_messages)

    started_at = perf_counter()
    outputs = llm.chat(chat_messages, sampling_params=build_sampling_params(INTERPRET_MAX_NEW_TOKENS))
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    request_output = outputs[0]
    result = request_output_text(request_output)
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    input_token_count = len(prompt_token_ids) if prompt_token_ids is not None else None
    output_token_count = request_output_token_count(request_output)
    log_resource_snapshot(
        "text_model.perf",
        torch_module=torch,
        extra={
            "elapsed_ms": elapsed_ms,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
        },
    )
    log_event("text_model.output", payload=result)
    return result


def run_vision_model(image_path: str, user_text: str, mode: str) -> str:
    llm = get_llm()
    resolved_image_path = Path(image_path).expanduser().resolve()
    log_event(
        "vision_model.request",
        payload={
            "image_path": str(resolved_image_path),
            "prompt": user_text,
        },
    )

    with Image.open(resolved_image_path) as handle:
        image = handle.convert("RGB").copy()

    chat_messages = [
        {"role": "system", "content": SYSTEM_TEXT_GROUND_V2},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    started_at = perf_counter()
    outputs = llm.chat(chat_messages, sampling_params=build_sampling_params(resolve_ground_max_new_tokens(mode)))
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    request_output = outputs[0]
    result = request_output_text(request_output)
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    input_token_count = len(prompt_token_ids) if prompt_token_ids is not None else None
    output_token_count = request_output_token_count(request_output)
    log_resource_snapshot(
        "vision_model.perf",
        torch_module=torch,
        extra={
            "elapsed_ms": elapsed_ms,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "mode": mode,
            "image_count": 1,
            "video_count": 0,
        },
    )
    log_event("vision_model.output", payload=result)
    return result


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


def ground_action(image_path: str, mode: str, task_spec: dict[str, Any], max_retries: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    last_raw_text: str | None = None
    retry_feedback = ""

    for attempt in range(1, max_retries + 1):
        prompt = build_ground_prompt_v2(mode, task_spec, NORMALIZED_COORD_MAX)
        if retry_feedback:
            prompt += (
                "\n\nPrevious output was invalid.\n"
                f"Validation error: {retry_feedback}\n"
                "Return valid JSON only."
            )

        log_event(
            "ground.request",
            f"attempt {attempt}/{max_retries}",
            {
                "image_path": image_path,
                "task_spec": task_spec,
                "prompt": prompt,
            },
        )

        try:
            raw_text = run_vision_model(image_path, prompt, mode)
            last_raw_text = raw_text
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
        except Exception as exc:
            last_error = exc
            retry_feedback = str(exc)
            log_retry("ground", attempt, max_retries, exc, last_raw_text)

    raise RuntimeError(
        f"Grounding failed after {max_retries} attempts. "
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


def resolve_input_text(task_spec: dict[str, Any], fallback_question: str) -> str:
    if task_spec["generate_random_text"]:
        length = task_spec["random_length"]
        if length is None:
            length = extract_requested_input_length(fallback_question) or DEFAULT_RANDOM_TEXT_LENGTH
        generated = random_text(length)
        log_event(
            "input.generated_text",
            payload={"length": length, "text": generated},
        )
        return generated
    resolved = clean_single_line_text(task_spec["input_text"])
    log_event("input.literal_text", payload=resolved)
    return resolved


def ask_image_json(image_path: str, question: str, mode: str, max_retries: int = 2) -> dict[str, Any]:
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

    task_spec = interpret_command(question, mode, run_text_model, max_retries=max_retries)
    log_event("ask.interpreted", payload=task_spec)

    if task_spec["status"] != "ok":
        if mode == "value":
            result = action_result(mode, "ambiguous", "", task_spec["comment"])
        elif mode == "multi_value":
            result = action_result(mode, "ambiguous", [], task_spec["comment"])
        else:
            result = action_result(mode, "ambiguous", None, task_spec["comment"])
        log_event("ask.result", payload=result)
        return result

    grounded = ground_action(image_path, mode, task_spec, max_retries=max_retries)

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
        text_to_type = resolve_input_text(task_spec, question)
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
    results_path = output_dir / f"qwen_vllm_results_{run_timestamp}.jsonl"

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
        ("point", "Return the center point of the Upload button"),
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
            draw(result["answer"], img, output_dir / f"qwen_vllm_{run_timestamp}_q{i}.png")
    print(json.dumps({"results_file": str(results_path)}, ensure_ascii=False))
