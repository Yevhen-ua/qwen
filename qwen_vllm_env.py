import json
import os
from pathlib import Path
from typing import Any


def load_model_parallel_constraints(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    text_config = config.get("text_config", {})
    vision_config = config.get("vision_config", {})

    constraints: dict[str, Any] = {"config_path": str(config_path)}

    attention_heads = text_config.get("num_attention_heads")
    if isinstance(attention_heads, int) and attention_heads > 0:
        constraints["num_attention_heads"] = attention_heads

    key_value_heads = text_config.get("num_key_value_heads")
    if isinstance(key_value_heads, int) and key_value_heads > 0:
        constraints["num_key_value_heads"] = key_value_heads

    vision_heads = vision_config.get("num_heads")
    if isinstance(vision_heads, int) and vision_heads > 0:
        constraints["vision_num_heads"] = vision_heads

    return constraints


def tensor_parallel_is_compatible(tp_size: int, constraints: dict[str, Any]) -> bool:
    if tp_size < 1:
        return False

    for key in ("num_attention_heads", "num_key_value_heads"):
        value = constraints.get(key)
        if isinstance(value, int) and value % tp_size != 0:
            return False
    return True


def resolve_compatible_tensor_parallel_size(requested_tp_size: int, constraints: dict[str, Any]) -> int:
    if tensor_parallel_is_compatible(requested_tp_size, constraints):
        return requested_tp_size

    for candidate in range(requested_tp_size - 1, 0, -1):
        if tensor_parallel_is_compatible(candidate, constraints):
            return candidate
    return 1


def detect_system_memory_gb() -> float | None:
    if hasattr(os, "sysconf"):
        if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
                return round((page_size * phys_pages) / (1024**3), 2)

    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MemoryStatus()
        memory_status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return round(memory_status.ullTotalPhys / (1024**3), 2)

    return None


def build_auto_defaults(
    min_gpu_memory_gb: float,
    visible_gpu_count: int,
    system_memory_gb: float | None,
    profile: str,
) -> tuple[dict[str, int | float], str]:
    profile_matrix: dict[str, dict[str, dict[str, int | float]]] = {
        "small": {
            "safe": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 1024,
                "max_num_seqs": 1,
                "cpu_offload_gb": 10.0,
                "mm_processor_cache_gb": 0.0,
            },
            "balanced": {
                "gpu_memory_utilization": 0.82,
                "max_model_len": 1536,
                "max_num_seqs": 1,
                "cpu_offload_gb": 8.0,
                "mm_processor_cache_gb": 0.0,
            },
            "aggressive": {
                "gpu_memory_utilization": 0.85,
                "max_model_len": 2048,
                "max_num_seqs": 1,
                "cpu_offload_gb": 6.0,
                "mm_processor_cache_gb": 0.0,
            },
        },
        "medium": {
            "safe": {
                "gpu_memory_utilization": 0.82,
                "max_model_len": 1536,
                "max_num_seqs": 1,
                "cpu_offload_gb": 6.0,
                "mm_processor_cache_gb": 0.0,
            },
            "balanced": {
                "gpu_memory_utilization": 0.85,
                "max_model_len": 2048,
                "max_num_seqs": 1,
                "cpu_offload_gb": 4.0,
                "mm_processor_cache_gb": 0.0,
            },
            "aggressive": {
                "gpu_memory_utilization": 0.88,
                "max_model_len": 3072,
                "max_num_seqs": 1,
                "cpu_offload_gb": 2.0,
                "mm_processor_cache_gb": 0.0,
            },
        },
        "large": {
            "safe": {
                "gpu_memory_utilization": 0.88,
                "max_model_len": 8192,
                "max_num_seqs": 1,
                "cpu_offload_gb": 2.0,
                "mm_processor_cache_gb": 2.0,
            },
            "balanced": {
                "gpu_memory_utilization": 0.92,
                "max_model_len": 12288,
                "max_num_seqs": 2,
                "cpu_offload_gb": 0.0,
                "mm_processor_cache_gb": 4.0,
            },
            "aggressive": {
                "gpu_memory_utilization": 0.95,
                "max_model_len": 16384,
                "max_num_seqs": 4,
                "cpu_offload_gb": 0.0,
                "mm_processor_cache_gb": 6.0,
            },
        },
    }

    if min_gpu_memory_gb <= 12:
        resource_bucket = "small"
    elif min_gpu_memory_gb <= 24:
        resource_bucket = "medium"
    else:
        resource_bucket = "large"

    settings = dict(profile_matrix[resource_bucket][profile])

    if system_memory_gb is None:
        settings["cpu_offload_gb"] = round(float(settings["cpu_offload_gb"]), 2)
    else:
        reserve_gb = 16.0 if system_memory_gb >= 64 else max(4.0, round(system_memory_gb * 0.2, 2))
        per_gpu_budget_gb = max((system_memory_gb - reserve_gb) / max(visible_gpu_count, 1), 0.0)
        settings["cpu_offload_gb"] = round(min(float(settings["cpu_offload_gb"]), per_gpu_budget_gb), 2)

    return settings, resource_bucket


def build_vllm_runtime_config() -> dict[str, Any]:
    profile = os.environ.get("QWEN_VLLM_PROFILE", "balanced").strip().lower() or "balanced"
    if profile not in {"safe", "balanced", "aggressive"}:
        raise ValueError("QWEN_VLLM_PROFILE must be one of: safe, balanced, aggressive")

    model_path = Path(os.environ.get("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")).expanduser()

    configured_devices = os.environ.get("QWEN_VLLM_CUDA_DEVICES", "").strip()
    if configured_devices:
        devices = [item.strip() for item in configured_devices.split(",") if item.strip()]
        if not devices:
            raise ValueError("QWEN_VLLM_CUDA_DEVICES must contain at least one device")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_device_count = len([item.strip() for item in visible_devices.split(",") if item.strip()]) if visible_devices else None

    configured_tp = os.environ.get("QWEN_VLLM_TENSOR_PARALLEL_SIZE", "").strip()
    model_parallel_constraints = load_model_parallel_constraints(model_path)
    if configured_tp:
        tensor_parallel_size = int(configured_tp)
        if tensor_parallel_size < 1:
            raise ValueError("QWEN_VLLM_TENSOR_PARALLEL_SIZE must be >= 1")
        if visible_device_count is not None and tensor_parallel_size > visible_device_count:
            raise ValueError("QWEN_VLLM_TENSOR_PARALLEL_SIZE cannot exceed the number of visible CUDA devices")
        if not tensor_parallel_is_compatible(tensor_parallel_size, model_parallel_constraints):
            raise ValueError("QWEN_VLLM_TENSOR_PARALLEL_SIZE is incompatible with the model attention head layout")
    else:
        tensor_parallel_size = None

    import torch

    visible_gpu_count = torch.cuda.device_count()
    if tensor_parallel_size is None:
        requested_tp_size = visible_gpu_count if visible_gpu_count > 0 else 1
        tensor_parallel_size = resolve_compatible_tensor_parallel_size(requested_tp_size, model_parallel_constraints)

    gpu_memories_gb: list[float] = []
    gpu_compute_capabilities: list[tuple[int, int]] = []
    if visible_gpu_count > 0:
        gpu_memories_gb = [
            round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2)
            for index in range(visible_gpu_count)
        ]
        gpu_compute_capabilities = [torch.cuda.get_device_capability(index) for index in range(visible_gpu_count)]

    system_memory_gb = detect_system_memory_gb()

    if gpu_memories_gb:
        min_gpu_memory_gb = min(gpu_memories_gb)
        recommended, resource_bucket = build_auto_defaults(
            min_gpu_memory_gb=min_gpu_memory_gb,
            visible_gpu_count=visible_gpu_count,
            system_memory_gb=system_memory_gb,
            profile=profile,
        )
    else:
        min_gpu_memory_gb = None
        resource_bucket = None
        recommended = {
            "gpu_memory_utilization": 0.9,
            "cpu_offload_gb": 0.0,
            "max_model_len": 8192,
            "max_num_seqs": 1,
            "mm_processor_cache_gb": 0.0,
        }

    if gpu_compute_capabilities:
        min_major = min(capability[0] for capability in gpu_compute_capabilities)
        recommended_dtype = "bfloat16" if min_major >= 8 else "float16"
    else:
        recommended_dtype = "auto"

    raw_dtype = os.environ.get("QWEN_VLLM_DTYPE", "").strip()
    dtype = raw_dtype or recommended_dtype

    raw_gpu_memory_utilization = os.environ.get("QWEN_VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    gpu_memory_utilization = (
        float(raw_gpu_memory_utilization)
        if raw_gpu_memory_utilization
        else float(recommended["gpu_memory_utilization"])
    )

    raw_cpu_offload_gb = os.environ.get("QWEN_VLLM_CPU_OFFLOAD_GB", "").strip()
    cpu_offload_gb = float(raw_cpu_offload_gb) if raw_cpu_offload_gb else float(recommended["cpu_offload_gb"])

    raw_max_model_len = os.environ.get("QWEN_VLLM_MAX_MODEL_LEN", "").strip()
    max_model_len = int(raw_max_model_len) if raw_max_model_len else int(recommended["max_model_len"])

    raw_max_num_seqs = os.environ.get("QWEN_VLLM_MAX_NUM_SEQS", "").strip()
    max_num_seqs = int(raw_max_num_seqs) if raw_max_num_seqs else int(recommended["max_num_seqs"])

    raw_limit_mm_images = os.environ.get("QWEN_VLLM_LIMIT_MM_IMAGES", "").strip()
    limit_mm_images = int(raw_limit_mm_images) if raw_limit_mm_images else 1

    raw_limit_mm_videos = os.environ.get("QWEN_VLLM_LIMIT_MM_VIDEOS", "").strip()
    limit_mm_videos = int(raw_limit_mm_videos) if raw_limit_mm_videos else 0

    raw_mm_processor_cache_gb = os.environ.get("QWEN_VLLM_MM_PROCESSOR_CACHE_GB", "").strip()
    mm_processor_cache_gb = (
        float(raw_mm_processor_cache_gb)
        if raw_mm_processor_cache_gb
        else float(recommended["mm_processor_cache_gb"])
    )

    raw_mm_processor_cache_type = os.environ.get("QWEN_VLLM_MM_PROCESSOR_CACHE_TYPE", "").strip()
    if raw_mm_processor_cache_type:
        mm_processor_cache_type = raw_mm_processor_cache_type
    elif tensor_parallel_size > 1:
        mm_processor_cache_type = "shm"
    else:
        mm_processor_cache_type = None

    raw_mm_encoder_tp_mode = os.environ.get("QWEN_VLLM_MM_ENCODER_TP_MODE", "").strip()
    mm_encoder_tp_mode = raw_mm_encoder_tp_mode or None

    return {
        "model_path": str(model_path),
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "cpu_offload_gb": cpu_offload_gb,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "limit_mm_per_prompt": {
            "image": limit_mm_images,
            "video": limit_mm_videos,
        },
        "mm_processor_cache_gb": mm_processor_cache_gb,
        "mm_processor_cache_type": mm_processor_cache_type,
        "mm_encoder_tp_mode": mm_encoder_tp_mode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") or None,
        "auto_defaults": {
            "profile": profile,
            "visible_gpu_count": visible_gpu_count,
            "gpu_memories_gb": gpu_memories_gb or None,
            "gpu_compute_capabilities": gpu_compute_capabilities or None,
            "min_gpu_memory_gb": min_gpu_memory_gb,
            "system_memory_gb": system_memory_gb,
            "resource_bucket": resource_bucket,
            "model_parallel_constraints": model_parallel_constraints or None,
            "resolved_tensor_parallel_size": tensor_parallel_size,
        },
    }


def save_vllm_runtime_config(config: dict[str, Any], output_path: Path) -> Path:
    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return resolved_path


if __name__ == "__main__":
    config = build_vllm_runtime_config()
    output_path = Path("qwen_vllm_runtime_config.json")
    output_path = save_vllm_runtime_config(config, output_path)
    print(json.dumps({"config_path": str(output_path), "config": config}, ensure_ascii=False, indent=2))
