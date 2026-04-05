import argparse
import base64
import json
import mimetypes
import socket
import subprocess
import time
from pathlib import Path

import requests


DEFAULT_MODEL = "ui-grounder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an SSH tunnel and send one OpenAI-compatible multimodal request.")
    parser.add_argument("--ssh-target", required=True, help="SSH target, for example user@cluster.")
    parser.add_argument("--remote-host", default="127.0.0.1", help="Remote host where the API server listens.")
    parser.add_argument("--remote-port", type=int, default=8000, help="Remote API server port.")
    parser.add_argument("--local-port", type=int, default=8000, help="Local forwarded port.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to send in the request body.")
    parser.add_argument("--mode", required=True, help="Inference mode: yes_no, point, input, drag, value, multi_value.")
    parser.add_argument("--question", required=True, help="Free-form user request.")
    parser.add_argument("--image", required=True, help="Path to a local image file.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds.")
    parser.add_argument("--tunnel-timeout", type=float, default=15.0, help="Timeout for establishing the SSH tunnel.")
    return parser.parse_args()


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def wait_for_local_port(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for local port {port} to accept connections")


def build_payload(args: argparse.Namespace, image_path: Path) -> dict:
    return {
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


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")

    ssh_command = [
        "ssh",
        "-L",
        f"{args.local_port}:{args.remote_host}:{args.remote_port}",
        "-N",
        args.ssh_target,
    ]
    tunnel = subprocess.Popen(ssh_command)
    try:
        wait_for_local_port(args.local_port, args.tunnel_timeout)
        api_url = f"http://127.0.0.1:{args.local_port}/v1/chat/completions"
        payload = build_payload(args, image_path)
        response = requests.post(api_url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait(timeout=5)


if __name__ == "__main__":
    main()
