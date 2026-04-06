import json
import os
import platform
import sys
from typing import Any


VERBOSE_LOGGING_ENABLED = os.environ.get("QWEN_VERBOSE_LOGGING", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
MAX_LOG_TEXT_CHARS = int(os.environ.get("QWEN_MAX_LOG_TEXT_CHARS", "4000"))


def _truncate_text(text: str, max_chars: int = MAX_LOG_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _format_payload(payload: Any) -> str:
    if isinstance(payload, dict) and "hf_device_map" in payload:
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return repr(payload)
    if isinstance(payload, str):
        return _truncate_text(payload)
    try:
        return _truncate_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        return _truncate_text(repr(payload))


def log_event(label: str, message: str | None = None, payload: Any | None = None) -> None:
    if not VERBOSE_LOGGING_ENABLED:
        return

    header = f"[qwen:{label}]"
    if message:
        print(f"{header} {message}", file=sys.stderr, flush=True)
    else:
        print(header, file=sys.stderr, flush=True)

    if payload is not None:
        print(_format_payload(payload), file=sys.stderr, flush=True)


def log_retry(
    label: str,
    attempt: int,
    max_retries: int,
    error: Exception,
    raw_text: str | None = None,
) -> None:
    if not VERBOSE_LOGGING_ENABLED:
        return

    print(
        f"[qwen:{label}] retry {attempt}/{max_retries} because: {error}",
        file=sys.stderr,
        flush=True,
    )
    if raw_text is not None:
        print(_format_payload(raw_text), file=sys.stderr, flush=True)


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


def _read_linux_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw_value = line.partition(":")
                parts = raw_value.strip().split()
                if not parts:
                    continue
                kib_value = int(parts[0])
                result[key] = kib_value * 1024
    except Exception:
        return {}
    return result


def _get_host_memory_snapshot() -> dict[str, Any]:
    meminfo = _read_linux_meminfo()
    total_bytes = meminfo.get("MemTotal")
    available_bytes = meminfo.get("MemAvailable")

    if total_bytes is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * phys_pages
        except Exception:
            total_bytes = None

    payload: dict[str, Any] = {
        "total_bytes": total_bytes,
        "total_gib": _bytes_to_gib(total_bytes),
        "available_bytes": available_bytes,
        "available_gib": _bytes_to_gib(available_bytes),
    }
    return payload


def _get_cuda_mem_info(torch_module: Any, index: int) -> tuple[int | None, int | None]:
    mem_get_info = getattr(torch_module.cuda, "mem_get_info", None)
    if mem_get_info is None:
        return None, None

    try:
        free_bytes, total_bytes = mem_get_info(index)
        return int(free_bytes), int(total_bytes)
    except Exception:
        pass

    current_device = None
    try:
        current_device = torch_module.cuda.current_device()
    except Exception:
        current_device = None

    try:
        torch_module.cuda.set_device(index)
        free_bytes, total_bytes = mem_get_info()
        return int(free_bytes), int(total_bytes)
    except Exception:
        return None, None
    finally:
        if current_device is not None:
            try:
                torch_module.cuda.set_device(current_device)
            except Exception:
                pass


def _get_gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        is_available = bool(torch_module.cuda.is_available())
    except Exception:
        is_available = False

    payload["is_available"] = is_available
    if not is_available:
        return payload

    try:
        device_count = int(torch_module.cuda.device_count())
    except Exception:
        device_count = 0

    payload["device_count"] = device_count
    devices: list[dict[str, Any]] = []

    for index in range(device_count):
        device_payload: dict[str, Any] = {"index": index}
        try:
            device_payload["name"] = torch_module.cuda.get_device_name(index)
        except Exception:
            device_payload["name"] = None

        try:
            props = torch_module.cuda.get_device_properties(index)
            total_memory_bytes = int(props.total_memory)
        except Exception:
            total_memory_bytes = None

        free_bytes, mem_get_info_total_bytes = _get_cuda_mem_info(torch_module, index)
        device_payload["total_memory_bytes"] = total_memory_bytes
        device_payload["total_memory_gib"] = _bytes_to_gib(total_memory_bytes)
        device_payload["free_memory_bytes"] = free_bytes
        device_payload["free_memory_gib"] = _bytes_to_gib(free_bytes)
        device_payload["mem_get_info_total_bytes"] = mem_get_info_total_bytes

        try:
            allocated_bytes = int(torch_module.cuda.memory_allocated(index))
            reserved_bytes = int(torch_module.cuda.memory_reserved(index))
        except Exception:
            allocated_bytes = None
            reserved_bytes = None

        device_payload["allocated_bytes"] = allocated_bytes
        device_payload["allocated_gib"] = _bytes_to_gib(allocated_bytes)
        device_payload["reserved_bytes"] = reserved_bytes
        device_payload["reserved_gib"] = _bytes_to_gib(reserved_bytes)
        devices.append(device_payload)

    payload["devices"] = devices
    return payload


def get_resource_snapshot(torch_module: Any | None = None, model: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "host_memory": _get_host_memory_snapshot(),
    }

    if torch_module is not None:
        payload["gpu"] = _get_gpu_snapshot(torch_module)

    if model is not None:
        payload["model_device"] = str(getattr(model, "device", None))
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is not None:
            payload["hf_device_map"] = hf_device_map

    return payload


def log_resource_snapshot(
    label: str,
    torch_module: Any | None = None,
    model: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = get_resource_snapshot(torch_module=torch_module, model=model)
    if extra:
        payload.update(extra)
    log_event(label, payload=payload)
