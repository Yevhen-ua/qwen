import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from vllm import LLM, SamplingParams

from qwen_logging import log_event, log_resource_snapshot
from qwen_parser7 import extract_json
from qwen_prompts import SYSTEM_TEXT_GROUND_V2
from qwen_schemas import (
    validate_drag_output as validate_drag_output_schema,
    validate_exists_output as validate_exists_output_schema,
    validate_input_output as validate_input_output_schema,
    validate_multi_value_output as validate_multi_value_output_schema,
    validate_point_output as validate_point_output_schema,
    validate_value_output as validate_value_output_schema,
)
from raw_answer_point import draw

MODEL_PATH = r"models\Qwen3-VL-8B-Instruct"
OUTPUT_DIR = Path("test_images/output")

GROUND_MAX_NEW_TOKENS_BY_MODE = {
    "yes_no": 128,
    "point": 128,
    "input": 128,
    "drag": 256,
    "value": 160,
    "multi_value": 512,
}
NORMALIZED_COORD_MAX = 1000

VLLM_TRUST_REMOTE_CODE = True
VLLM_TENSOR_PARALLEL_SIZE = 4
VLLM_DTYPE = "auto"
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_CPU_OFFLOAD_GB = 0.0
VLLM_MAX_MODEL_LEN = 8192
VLLM_MAX_NUM_SEQS = 1
VLLM_LIMIT_MM_IMAGES = 1
VLLM_LIMIT_MM_VIDEOS = 0
VLLM_ENABLE_THINKING = False

llm_kwargs = {
    "model": MODEL_PATH,
    "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
    "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
    "dtype": VLLM_DTYPE,
    "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
    "max_model_len": VLLM_MAX_MODEL_LEN,
    "max_num_seqs": VLLM_MAX_NUM_SEQS,
    "limit_mm_per_prompt": {
        "image": VLLM_LIMIT_MM_IMAGES,
        "video": VLLM_LIMIT_MM_VIDEOS,
    },
}

_llm = None
_runtime_logged = False


def get_llm():
    global _llm, _runtime_logged
    if _llm is None:
        _llm = LLM(**llm_kwargs)

    if not _runtime_logged:
        log_event(
            "runtime.config",
            payload={
                "model_path": MODEL_PATH,
                "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
                "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
                "dtype": VLLM_DTYPE,
                "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
                "cpu_offload_gb": None,
                "max_model_len": VLLM_MAX_MODEL_LEN,
                "max_num_seqs": VLLM_MAX_NUM_SEQS,
                "limit_mm_per_prompt": {
                    "image": VLLM_LIMIT_MM_IMAGES,
                    "video": VLLM_LIMIT_MM_VIDEOS,
                },
                "enable_thinking": VLLM_ENABLE_THINKING,
            },
        )
        log_resource_snapshot("runtime.resources", torch_module=torch)
        _runtime_logged = True

    return _llm


def normalize_mode(mode):
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
    return aliases[value]


def get_output_dir():
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_result_record(output_path, payload):
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_sampling_params(max_tokens):
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )


def request_output_text(request_output):
    if not getattr(request_output, "outputs", None):
        raise RuntimeError("vLLM returned no outputs")
    return str(request_output.outputs[0].text).strip()


def request_output_token_count(request_output):
    if not getattr(request_output, "outputs", None):
        return None
    token_ids = getattr(request_output.outputs[0], "token_ids", None)
    if token_ids is None:
        return None
    return len(token_ids)


def run_vision_model(image_path, user_text, mode):
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
    outputs = llm.chat(
        chat_messages,
        chat_template_kwargs={"enable_thinking": VLLM_ENABLE_THINKING},
        sampling_params=build_sampling_params(GROUND_MAX_NEW_TOKENS_BY_MODE[mode]),
    )
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
            "video_count": VLLM_LIMIT_MM_VIDEOS,
        },
    )
    log_event("vision_model.output", payload=result)
    return result


def build_direct_ground_prompt(mode, question):
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
            "- text: string; the exact text that should be typed based on the user request\n"
            "- comment: string; brief reason why the coordinates and text were returned\n"
            "Rules:\n"
            "- identify the editable field the user wants to type into\n"
            "- determine the text to type directly from the free-form user request\n"
            "- if the request asks for generated or random text, choose it yourself and return the exact chosen text\n"
            "- if the request does not specify any text to type, return text as an empty string\n"
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

    raise KeyError(mode)


def validate_exists_output(data):
    validated = validate_exists_output_schema(data)
    return {
        "status": validated["status"],
        "comment": validated["comment"],
    }


def validate_point_output(data):
    validated = validate_point_output_schema(data, NORMALIZED_COORD_MAX)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "comment": validated["comment"],
    }


def validate_input_output(data):
    validated = validate_input_output_schema(data, NORMALIZED_COORD_MAX)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "text": validated["text"],
        "comment": validated["comment"],
    }


def validate_value_output(data):
    validated = validate_value_output_schema(data)
    return {
        "status": validated["status"],
        "answer": validated["answer"],
        "comment": validated["comment"],
    }


def validate_multi_value_output(data):
    validated = validate_multi_value_output_schema(data)
    return {
        "status": validated["status"],
        "answer": validated["answer"],
        "comment": validated["comment"],
    }


def validate_drag_output(data):
    validated = validate_drag_output_schema(data, NORMALIZED_COORD_MAX)
    return {
        "status": validated["status"],
        "x": validated["x"],
        "y": validated["y"],
        "x2": validated["x2"],
        "y2": validated["y2"],
        "comment": validated["comment"],
    }


def ground_action(image_path, question, mode):
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
    elif mode == "point":
        validated = validate_point_output(data)
    elif mode == "input":
        validated = validate_input_output(data)
    elif mode == "value":
        validated = validate_value_output(data)
    elif mode == "multi_value":
        validated = validate_multi_value_output(data)
    elif mode == "drag":
        validated = validate_drag_output(data)
    else:
        raise KeyError(mode)
    log_event("ground.result", payload=validated)
    return validated


def normalized_to_pixels(x, y, image_path):
    width, height = Image.open(image_path).size
    pixel_x = round(x * (width - 1) / NORMALIZED_COORD_MAX)
    pixel_y = round(y * (height - 1) / NORMALIZED_COORD_MAX)
    return {"x": pixel_x, "y": pixel_y}


def normalized_drag_to_pixels(x, y, x2, y2, image_path):
    width, height = Image.open(image_path).size
    return {
        "x": round(x * (width - 1) / NORMALIZED_COORD_MAX),
        "y": round(y * (height - 1) / NORMALIZED_COORD_MAX),
        "x2": round(x2 * (width - 1) / NORMALIZED_COORD_MAX),
        "y2": round(y2 * (height - 1) / NORMALIZED_COORD_MAX),
    }


def null_xy():
    return {"x": None, "y": None}


def null_drag():
    return {"x": None, "y": None, "x2": None, "y2": None}


def action_result(mode, status, answer, comment):
    return {
        "mode": mode,
        "status": status,
        "answer": answer,
        "comment": comment,
    }


def ask_image_json(image_path, question, mode):
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
        answer = {"x": None, "y": None, "text": grounded["text"]}
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

    raise KeyError(mode)


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
