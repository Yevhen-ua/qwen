# Apptainer container for this project

Inside the image:

- Ubuntu 22.04
- System Python 3.10
- PyTorch CUDA 12.6
- `transformers 4.57.6`

## Build

```bash
apptainer build qwen-cu126-py310.sif apptainer/qwen-cu126-py310.def
```

## Notes

- The image uses Ubuntu 22.04 system Python `3.10` and the official PyTorch `cu126` wheels.
- `flash-attn` is intentionally not installed. Your code already falls back to `sdpa`.
- The container sets only `HF_HOME=/hf-cache` and `TRANSFORMERS_CACHE=/hf-cache` by default.
- Set `QWEN_MODEL_PATH` and `QWEN_OUTPUT_DIR` explicitly at launch time if you need custom paths.
