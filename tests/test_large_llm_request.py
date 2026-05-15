import math
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from llm_connector import encode_image_as_data_url
from prompts import SYSTEM_TEXT

BASE_DIR = Path(__file__).resolve().parents[1]
APPROX_IMG_TOKENS = 900
TARGET_TOKENS = 30000
CHARS_PER_TOKEN = 3.0
MAX_RESPONSE_TOKENS = 500
TEMPERATURE = 0.0


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(math.ceil(len(text) / chars_per_token))


def trim_text_to_char_budget(text: str, char_budget: int) -> str:
    if len(text) <= char_budget:
        return text

    trimmed = text[:char_budget]
    last_newline = trimmed.rfind("\n")
    if last_newline > 0:
        trimmed = trimmed[:last_newline]
    return trimmed


def build_static_text() -> str:
    return (
        "Large multimodal test request.\n"
        "Use the attached screenshot as the primary visual source.\n"
        "The DOM dump below is auxiliary context and may include invisible or stale nodes.\n"
        "Return exactly one JSON object with these keys:\n"
        '- "page_title": string\n'
        '- "main_brand": string\n'
        '- "visible_search_placeholder": string\n'
        '- "top_actions": array of strings\n'
        '- "short_comment": string\n'
        "Do not include markdown fences.\n"
        "\n"
        "DOM snapshot excerpt follows.\n"
    )


def fill_prompt(tokens_cnt: int) -> str:
    chunk = (
        "garbage prompt filler token block lorem ipsum synthetic context "
        "repeated noise sequence for load testing only "
    )
    parts = []
    while estimate_tokens("".join(parts), CHARS_PER_TOKEN) < tokens_cnt:
        parts.append(chunk)
    return "".join(parts)


def build_user_text(dom_text: str, target_tokens: int, chars_per_token: float, prompt_name: str) -> str:
    static_text = (
        f"request_name: {prompt_name}\n"
        f"request_timestamp: {datetime.now(UTC).isoformat()}\n"
        f"request_uuid: {uuid.uuid4()}\n\n"
        f"{build_static_text()}"
    )
    target_text_tokens = max(target_tokens - APPROX_IMG_TOKENS, 0)
    target_chars = int(target_text_tokens * chars_per_token)
    dom_char_budget = max(target_chars - len(static_text), 0)
    dom_excerpt = trim_text_to_char_budget(dom_text, dom_char_budget)
    user_text = f"{static_text}\n```html\n{dom_excerpt}\n```"
    missing_tokens = target_text_tokens - estimate_tokens(user_text, chars_per_token)
    if missing_tokens > 0:
        user_text = f"{user_text}\n\n{fill_prompt(missing_tokens)}"
    return user_text


def run_prompt(llmconnector, prompt_name: str, image_name: str, dom_name: str) -> dict:
    image_path = BASE_DIR / "test_images" / "input" / image_name
    dom_path = BASE_DIR / "test_images" / "input" / dom_name
    dom_text = dom_path.read_text(encoding="utf-8")
    user_text = build_user_text(
        dom_text=dom_text,
        target_tokens=TARGET_TOKENS,
        chars_per_token=CHARS_PER_TOKEN,
        prompt_name=prompt_name,
    )

    payload = {
        "model": llmconnector.model,
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_as_data_url(image_path)},
                    },
                ],
            },
        ],
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
    }

    response = json.loads(llmconnector.request(payload))
    print(response.get("usage"))
    return response


def test_prompt_1(llmconnector):
    response = run_prompt(llmconnector, "prompt_1", "roz_test_01.png", "roz_test_01_data")

    assert isinstance(response, dict)


def test_prompt_2(llmconnector):
    response = run_prompt(llmconnector, "prompt_2", "roz_test_02.png", "roz_test_02_data")

    assert isinstance(response, dict)


def test_prompt_3(llmconnector):
    response = run_prompt(llmconnector, "prompt_3", "roz_test_03.png", "roz_test_03_data")

    assert isinstance(response, dict)


def test_prompt_4(llmconnector):
    response = run_prompt(llmconnector, "prompt_4", "roz_test_01.png", "roz_test_02_data")

    assert isinstance(response, dict)


def test_prompt_5(llmconnector):
    response = run_prompt(llmconnector, "prompt_5", "roz_test_01.png", "roz_test_03_data")

    assert isinstance(response, dict)


def test_prompt_6(llmconnector):
    response = run_prompt(llmconnector, "prompt_6", "roz_test_02.png", "roz_test_01_data")

    assert isinstance(response, dict)


def test_prompt_7(llmconnector):
    response = run_prompt(llmconnector, "prompt_7", "roz_test_02.png", "roz_test_03_data")

    assert isinstance(response, dict)


def test_prompt_8(llmconnector):
    response = run_prompt(llmconnector, "prompt_8", "roz_test_03.png", "roz_test_01_data")

    assert isinstance(response, dict)


def test_prompt_9(llmconnector):
    response = run_prompt(llmconnector, "prompt_9", "roz_test_03.png", "roz_test_02_data")

    assert isinstance(response, dict)


def test_prompt_10(llmconnector):
    response = run_prompt(llmconnector, "prompt_10", "roz_test_01.png", "roz_test_01_data")

    assert isinstance(response, dict)
