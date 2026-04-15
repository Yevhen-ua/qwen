import base64
import json
import mimetypes
from contextlib import contextmanager
from pathlib import Path

import requests

import config
from prompts import SYSTEM_TEXT, prom

DEFAULT_API_URL = f"http://127.0.0.1:{config.SSH_LOCAL_PORT}/v1/chat/completions"
DEFAULT_MODEL = "ui-grounder"


def encode_image_as_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def mode_selector(mode, question):
    return prom[mode].format(question=question)


class LlmConnector:
    def __init__(
        self,
        api_url=DEFAULT_API_URL,
        model=DEFAULT_MODEL,
        timeout=300.0,
    ):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self._tunnel = None

    @contextmanager
    def ssh_tunnel(self):
        if self._tunnel is not None:
            raise RuntimeError("SSH tunnel is already running")

        try:
            from sshtunnel import SSHTunnelForwarder
        except ImportError as exc:
            raise RuntimeError("Install sshtunnel to use LlmConnector.ssh_tunnel()") from exc

        tunnel_kwargs = {
            "ssh_address_or_host": (config.SSH_HOST, config.SSH_PORT),
            "remote_bind_address": (config.SSH_REMOTE_HOST, config.SSH_REMOTE_PORT),
            "local_bind_address": ("127.0.0.1", config.SSH_LOCAL_PORT),
            "set_keepalive": 30.0,
            "ssh_timeout": config.SSH_TIMEOUT,
            "tunnel_timeout": config.SSH_TIMEOUT,
            "ssh_username": config.SSH_USER,
            "ssh_password": config.SSH_PASS
        }

        self._tunnel = SSHTunnelForwarder(**tunnel_kwargs)
        previous_api_url = self.api_url

        try:
            self._tunnel.start()
            self.api_url = f"http://127.0.0.1:{self._tunnel.local_bind_port}/v1/chat/completions"
            yield self
        finally:
            self.api_url = previous_api_url
            tunnel = self._tunnel
            self._tunnel = None
            if tunnel is None:
                return
            tunnel.stop()

    def request(
        self,
        mode,
        question,
        image_path,
        max_tokens=None,
        temperature=None,
    ):
        resolved_image_path = Path(image_path)
        resolved_image_path = resolved_image_path.expanduser().resolve()

        built_question = mode_selector(mode, question)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_TEXT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": built_question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image_as_data_url(resolved_image_path),
                            },
                        },
                    ],
                }
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        response_payload = response.json()
        content = response_payload["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content

        return json.loads(content)
