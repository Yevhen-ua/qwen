import asyncio
import base64
import binascii
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from qwen_vllm2 import MODEL_PATH, ask_image_json, get_llm


MAX_IMAGE_BYTES = int(os.environ.get("QWEN_SERVER_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
EAGER_LOAD = os.environ.get("QWEN_SERVER_EAGER_LOAD", "").strip() == "1"


class InferRequest(BaseModel):
    mode: Literal["yes_no", "y_n", "yes/no", "yn", "point", "input", "drag", "value", "multi_value"]
    question: str = Field(min_length=1)
    image_base64: str = Field(min_length=1)
    request_id: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    mode: Literal["yes_no", "y_n", "yes/no", "yn", "point", "input", "drag", "value", "multi_value"]
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "loading"]
    model_loaded: bool


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.infer_lock = asyncio.Lock()
    app.state.model_loaded = False
    if EAGER_LOAD:
        await run_in_threadpool(get_llm)
        app.state.model_loaded = True
    yield


app = FastAPI(
    title="Qwen UI Grounding API",
    version="0.1.0",
    lifespan=lifespan,
)


def decode_image_bytes(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image_base64 decoded to empty payload")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"image payload exceeds {MAX_IMAGE_BYTES} bytes")
    return image_bytes


def write_temp_image(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as handle:
            image = handle.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="request image is not a supported image format") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
        image.save(handle, format="PNG")
        return handle.name


def extract_data_url_payload(url: str) -> str:
    if not url.startswith("data:"):
        raise HTTPException(status_code=400, detail="Only data:image/...;base64 URLs are supported")
    _, _, payload = url.partition(",")
    if not payload:
        raise HTTPException(status_code=400, detail="image_url data URL is missing base64 payload")
    return payload


def extract_openai_user_input(messages: list[ChatMessage]) -> tuple[str, str]:
    user_messages = [message for message in messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")

    message = user_messages[-1]
    if isinstance(message.content, str):
        raise HTTPException(
            status_code=400,
            detail="User message must include multimodal content with text and image_url",
        )

    question_parts: list[str] = []
    image_base64: str | None = None
    for item in message.content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", ""))
        if item_type == "text":
            text = str(item.get("text", "")).strip()
            if text:
                question_parts.append(text)
            continue
        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise HTTPException(status_code=400, detail="image_url must contain a string url")
            image_base64 = extract_data_url_payload(image_url)
            continue

    question = "\n".join(question_parts).strip()
    if not question:
        raise HTTPException(status_code=400, detail="User message must contain at least one text content item")
    if not image_base64:
        raise HTTPException(status_code=400, detail="User message must contain one image_url content item")
    return question, image_base64


def run_inference(request: InferRequest) -> dict[str, Any]:
    image_bytes = decode_image_bytes(request.image_base64)
    temp_image_path = write_temp_image(image_bytes)
    try:
        return ask_image_json(temp_image_path, request.question, request.mode)
    finally:
        Path(temp_image_path).unlink(missing_ok=True)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=ReadyResponse)
async def readyz() -> ReadyResponse:
    loaded = bool(app.state.model_loaded)
    return ReadyResponse(status="ready" if loaded else "loading", model_loaded=loaded)


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    return ModelsResponse(
        object="list",
        data=[
            {
                "id": "ui-grounder",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "root": MODEL_PATH,
            }
        ],
    )


@app.post("/infer")
async def infer(request: InferRequest) -> dict[str, Any]:
    started_at = perf_counter()
    async with app.state.infer_lock:
        try:
            result = await run_in_threadpool(run_inference, request)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    app.state.model_loaded = True
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    response: dict[str, Any] = {
        "result": result,
        "elapsed_ms": elapsed_ms,
    }
    if request.request_id is not None:
        response["request_id"] = request.request_id
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionsRequest) -> dict[str, Any]:
    question, image_base64 = extract_openai_user_input(request.messages)
    infer_request = InferRequest(
        mode=request.mode,
        question=question,
        image_base64=image_base64,
    )

    started_at = perf_counter()
    async with app.state.infer_lock:
        try:
            result = await run_in_threadpool(run_inference, infer_request)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    app.state.model_loaded = True
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    result_text = json.dumps(result, ensure_ascii=False)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "service_timing": {
            "elapsed_ms": elapsed_ms,
        },
    }


# Example:
#   uvicorn qwen_rest_server:app --host 0.0.0.0 --port 8000 --workers 1
