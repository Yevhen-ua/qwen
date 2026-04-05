import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "ui-grounder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one OpenAI-compatible multimodal request to qwen_rest_server.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat Completions endpoint URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to send in the request body.")
    parser.add_argument("--mode", required=True, help="Inference mode: yes_no, point, input, drag, value, multi_value.")
    parser.add_argument("--question", required=True, help="Free-form user request.")
    parser.add_argument("--image", required=True, help="Path to a local image file.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")

    payload = {
        "model": args.model,
        "mode": args.mode,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image_as_data_url(image_path),
                        },
                    },
                ],
            }
        ],
    }

    response = requests.post(args.api_url, json=payload, timeout=args.timeout)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
