import json
import os
from pathlib import Path
from typing import Any


def parse_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_model_path() -> Path:
    configured_path = os.environ.get("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")
    return Path(configured_path).expanduser()


def load_model_parallel_constraints() -> dict[str, Any]:
    config_path = get_model_path() / "config.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    text_config = config.get("text_config", {})
    vision_config = config.get("vision_config", {})

    constraints: dict[str, Any] = {
        "config_path": str(config_path),
    }

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


def resolve_compatible_tensor_parallel_size(
    requested_tp_size: int,
    constraints: dict[str, Any],
) -> int:
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


def choose_resource_bucket(min_gpu_memory_gb: float) -> str:
    if min_gpu_memory_gb <= 12:
        return "small"
    if min_gpu_memory_gb <= 24:
        return "medium"
    return "large"


def resolve_dtype_for_devices(compute_capabilities: list[tuple[int, int]]) -> str:
    if not compute_capabilities:
        return "auto"

    min_major = min(capability[0] for capability in compute_capabilities)
    if min_major >= 8:
        return "bfloat16"
    return "float16"


def clamp_host_budget(target_gb: float, system_memory_gb: float | None, visible_gpu_count: int) -> float:
    if system_memory_gb is None:
        return round(target_gb, 2)

    reserve_gb = 16.0 if system_memory_gb >= 64 else max(4.0, round(system_memory_gb * 0.2, 2))
    per_gpu_budget_gb = max((system_memory_gb - reserve_gb) / max(visible_gpu_count, 1), 0.0)
    return round(min(target_gb, per_gpu_budget_gb), 2)


def set_default_env(
    env_name: str,
    value: int | float,
    auto_defaults: dict[str, Any],
    auto_key: str,
) -> None:
    if os.environ.get(env_name, "").strip():
        return

    if isinstance(value, float) and value.is_integer():
        rendered_value = str(int(value))
    else:
        rendered_value = str(value)
    os.environ[env_name] = rendered_value
    auto_defaults[auto_key] = value


def build_auto_defaults(
    min_gpu_memory_gb: float,
    visible_gpu_count: int,
    system_memory_gb: float | None,
    profile: str,
) -> dict[str, int | float]:
    profile_matrix: dict[str, dict[str, dict[str, int | float]]] = {
        "small": {
            "safe": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 1536,
                "max_num_seqs": 1,
                "cpu_offload_gb": 10.0,
                "mm_processor_cache_gb": 0.0,
            },
            "balanced": {
                "gpu_memory_utilization": 0.84,
                "max_model_len": 2048,
                "max_num_seqs": 1,
                "cpu_offload_gb": 8.0,
                "mm_processor_cache_gb": 0.0,
            },
            "aggressive": {
                "gpu_memory_utilization": 0.88,
                "max_model_len": 3072,
                "max_num_seqs": 1,
                "cpu_offload_gb": 6.0,
                "mm_processor_cache_gb": 1.0,
            },
        },
        "medium": {
            "safe": {
                "gpu_memory_utilization": 0.84,
                "max_model_len": 4096,
                "max_num_seqs": 1,
                "cpu_offload_gb": 6.0,
                "mm_processor_cache_gb": 1.0,
            },
            "balanced": {
                "gpu_memory_utilization": 0.88,
                "max_model_len": 6144,
                "max_num_seqs": 1,
                "cpu_offload_gb": 4.0,
                "mm_processor_cache_gb": 2.0,
            },
            "aggressive": {
                "gpu_memory_utilization": 0.92,
                "max_model_len": 8192,
                "max_num_seqs": 2,
                "cpu_offload_gb": 2.0,
                "mm_processor_cache_gb": 3.0,
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

    bucket = choose_resource_bucket(min_gpu_memory_gb)
    settings = dict(profile_matrix[bucket][profile])
    settings["cpu_offload_gb"] = clamp_host_budget(
        float(settings["cpu_offload_gb"]), system_memory_gb, visible_gpu_count
    )
    return settings


def bootstrap_vllm_environment() -> dict[str, Any]:
    profile = os.environ.get("QWEN_VLLM_PROFILE", "balanced").strip().lower() or "balanced"
    if profile not in {"safe", "balanced", "aggressive"}:
        raise ValueError("QWEN_VLLM_PROFILE must be one of: safe, balanced, aggressive")

    configured_devices = os.environ.get("QWEN_VLLM_CUDA_DEVICES", "").strip()
    if configured_devices:
        devices = parse_visible_devices(configured_devices)
        if not devices:
            raise ValueError("QWEN_VLLM_CUDA_DEVICES must contain at least one device")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_device_count = len(parse_visible_devices(visible_devices)) if visible_devices else None

    configured_tp = os.environ.get("QWEN_VLLM_TENSOR_PARALLEL_SIZE", "").strip()
    model_parallel_constraints = load_model_parallel_constraints()
    if configured_tp:
        tensor_parallel_size = int(configured_tp)
        if tensor_parallel_size < 1:
            raise ValueError("QWEN_VLLM_TENSOR_PARALLEL_SIZE must be >= 1")
        if visible_device_count is not None and tensor_parallel_size > visible_device_count:
            raise ValueError(
                "QWEN_VLLM_TENSOR_PARALLEL_SIZE cannot exceed the number of visible CUDA devices"
            )
        if not tensor_parallel_is_compatible(tensor_parallel_size, model_parallel_constraints):
            raise ValueError(
                "QWEN_VLLM_TENSOR_PARALLEL_SIZE is incompatible with the model attention head layout"
            )
    elif visible_device_count is not None and visible_device_count > 0:
        tensor_parallel_size = resolve_compatible_tensor_parallel_size(
            visible_device_count,
            model_parallel_constraints,
        )
        os.environ["QWEN_VLLM_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    else:
        tensor_parallel_size = None

    import torch

    if tensor_parallel_size is None:
        visible_gpu_count = torch.cuda.device_count()
        requested_tp_size = visible_gpu_count if visible_gpu_count > 0 else 1
        tensor_parallel_size = resolve_compatible_tensor_parallel_size(
            requested_tp_size,
            model_parallel_constraints,
        )
        os.environ["QWEN_VLLM_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)

    gpu_memories_gb: list[float] = []
    gpu_compute_capabilities: list[tuple[int, int]] = []
    if torch.cuda.device_count() > 0:
        gpu_memories_gb = [
            round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2)
            for index in range(torch.cuda.device_count())
        ]
        gpu_compute_capabilities = [
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        ]

    system_memory_gb = detect_system_memory_gb()
    auto_defaults: dict[str, Any] = {}
    if gpu_memories_gb:
        min_gpu_memory_gb = min(gpu_memories_gb)
        recommended = build_auto_defaults(
            min_gpu_memory_gb=min_gpu_memory_gb,
            visible_gpu_count=torch.cuda.device_count(),
            system_memory_gb=system_memory_gb,
            profile=profile,
        )

        set_default_env(
            "QWEN_VLLM_DTYPE",
            resolve_dtype_for_devices(gpu_compute_capabilities),
            auto_defaults,
            "dtype",
        )
        set_default_env(
            "QWEN_VLLM_GPU_MEMORY_UTILIZATION",
            float(recommended["gpu_memory_utilization"]),
            auto_defaults,
            "gpu_memory_utilization",
        )
        set_default_env(
            "QWEN_VLLM_CPU_OFFLOAD_GB",
            float(recommended["cpu_offload_gb"]),
            auto_defaults,
            "cpu_offload_gb",
        )
        set_default_env(
            "QWEN_VLLM_MAX_MODEL_LEN",
            int(recommended["max_model_len"]),
            auto_defaults,
            "max_model_len",
        )
        set_default_env(
            "QWEN_VLLM_MAX_NUM_SEQS",
            int(recommended["max_num_seqs"]),
            auto_defaults,
            "max_num_seqs",
        )
        set_default_env(
            "QWEN_VLLM_MM_PROCESSOR_CACHE_GB",
            float(recommended["mm_processor_cache_gb"]),
            auto_defaults,
            "mm_processor_cache_gb",
        )

        auto_defaults["profile"] = profile
        auto_defaults["visible_gpu_count"] = torch.cuda.device_count()
        auto_defaults["gpu_memories_gb"] = gpu_memories_gb
        auto_defaults["gpu_compute_capabilities"] = gpu_compute_capabilities
        auto_defaults["min_gpu_memory_gb"] = min_gpu_memory_gb
        auto_defaults["system_memory_gb"] = system_memory_gb
        auto_defaults["resource_bucket"] = choose_resource_bucket(min_gpu_memory_gb)
        auto_defaults["model_parallel_constraints"] = model_parallel_constraints or None
        auto_defaults["resolved_tensor_parallel_size"] = tensor_parallel_size

    return {
        "tensor_parallel_size": tensor_parallel_size,
        "auto_defaults": auto_defaults or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
