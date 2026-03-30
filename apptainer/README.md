# Apptainer base container

Inside the image:

- Ubuntu 22.04
- System Python 3.10
- PyTorch CUDA 12.6
- no project-specific Python dependencies

## Build

```bash
apptainer build qwen-cu126-py310.sif apptainer/qwen-cu126-py310.def
```

## Notes

- The image uses Ubuntu 22.04 system Python `3.10` and the official PyTorch `cu126` wheels.
- Install project dependencies separately in a mounted project directory.
- The container sets only `HF_HOME=/hf-cache` and `TRANSFORMERS_CACHE=/hf-cache` by default.
