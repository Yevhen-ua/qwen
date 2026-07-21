# Cluster Runbook

Short instructions for running an inference job on the cluster, downloading models from Hugging Face, and building Apptainer/Singularity images.

## 1. Download a Model from Hugging Face

The download is performed on the cluster through the local Artifactory mirror, not directly from the public internet. In this setup, `HF_TOKEN` must be an Artifactory token, not a Hugging Face token.

Download the full repository:

```bash
HF_TOKEN=<your_token> MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct sbatch cluster/hf_download.sbatch
```

Download one file:

```bash
HF_TOKEN=<your_token> MODEL_ID=repo/name HF_FILE=config.json sbatch cluster/hf_download.sbatch
```

Download multiple files:

```bash
HF_TOKEN=<your_token> MODEL_ID=repo/name HF_FILES="model.gguf mmproj.gguf" sbatch cluster/hf_download.sbatch
```

Set the target directory:

```bash
HF_TOKEN=<your_token> MODEL_ID=repo/name LOCAL_DIR=/workspace/models/my-model sbatch cluster/hf_download.sbatch
```

By default, `WORKSPACE` is `/projects/sec_qa`, which is mounted inside the container as `/workspace`. Files are written to:

```bash
/workspace/models/<MODEL_ID with "/" replaced by "__">
```

On the cluster filesystem, this corresponds to:

```bash
/projects/sec_qa/models/<MODEL_ID with "/" replaced by "__">
```

For example, `Qwen/Qwen2.5-VL-7B-Instruct` becomes:

```bash
/workspace/models/Qwen__Qwen2.5-VL-7B-Instruct
```

## 2. Build the Apptainer Image Locally

The cluster has no internet access and images cannot be built there. Build the `.sif` image locally on a machine with internet access, then copy the finished file to `/projects/sec_qa/containers` on the cluster.

For llama.cpp with CUDA 13.1 / Python 3.12 / AVX512:

```bash
cd <local_repo_path>
apptainer build containers/llama-cu131-py312-serve-avx512-sched1.sif apptainer/llama-cu131-py312-serve-avx512-sched1.def
```

For vLLM with CUDA 13.1 / Python 3.12:

```bash
cd <local_repo_path>
apptainer build containers/vllm0230-cu131-py312-serve.sif apptainer/vllm-cu131-py312-serve.def
```

There are two options for delivering the finished `.sif` image to the cluster.

Option 1: copy the image directly to the cluster:

```bash
scp containers/llama-cu131-py312-serve-avx512-sched1.sif <user>@<cluster>:/projects/sec_qa/containers/
scp containers/vllm0230-cu131-py312-serve.sif <user>@<cluster>:/projects/sec_qa/containers/
```

Option 2: upload the image to Artifactory from the local machine, then download it from Artifactory on the cluster. Replace the Artifactory URL with the correct one:

```bash
curl -H "Authorization: Bearer <artifactory_token>" \
  -T containers/llama-cu131-py312-serve-avx512-sched1.sif \
  "<artifactory_url>/llama-cu131-py312-serve-avx512-sched1.sif"

curl -H "Authorization: Bearer <artifactory_token>" \
  -T containers/vllm0230-cu131-py312-serve.sif \
  "<artifactory_url>/vllm0230-cu131-py312-serve.sif"
```

On the cluster:

```bash
cd /projects/sec_qa
curl -H "Authorization: Bearer <artifactory_token>" \
  -o containers/llama-cu131-py312-serve-avx512-sched1.sif \
  "<artifactory_url>/llama-cu131-py312-serve-avx512-sched1.sif"

curl -H "Authorization: Bearer <artifactory_token>" \
  -o containers/vllm0230-cu131-py312-serve.sif \
  "<artifactory_url>/vllm0230-cu131-py312-serve.sif"
```

## 3. Request 4 GPUs in SLURM

```bash
cd /projects/sec_qa
sbatch cluster/my_gpu_4.sbatch
```

Check the job state:

```bash
squeue -u $USER
```

Logs:

```bash
tail -f /projects/sec_qa/logs/my_gpu_4-<JOB_ID>.out
tail -f /projects/sec_qa/logs/my_gpu_4-<JOB_ID>.err
```

## 4. Enter the Active Job

After the job starts, get `JOB_ID` from `sbatch` or `squeue`:

```bash
srun --jobid=<JOB_ID> --overlap --pty bash -i
cd /projects/sec_qa
```

## 5. Start the Inference Server

`cluster/my_gpu_4.sbatch` creates and keeps the `inference` tmux session alive inside the SLURM job.

After entering the active job, attach to that tmux session:

```bash
tmux attach -t inference
```

### 5a. Start llama.cpp Router

Inside the tmux session, start the llama router:

```bash
cd /projects/sec_qa
bash cluster/llama_stack_4_router.sh
```

Run the router only inside the SLURM job, because the script uses `SLURM_JOB_ID`.

The script creates or reuses these tmux windows:

```text
shell   - regular shell session
llama   - llama-server on 127.0.0.1:55144
tunnel  - reverse SSH tunnel for port 55144
```

Important: `cluster/llama_stack_4_router.sh` currently points to this container:

```bash
containers/llama9489_avx-cu126-py310-serve.sif
```

If a different `.sif` is used, update the path in `SHELL_CMD` or name the new image accordingly.

### 5b. Start vLLM

Inside the tmux session, start vLLM:

```bash
cd /projects/sec_qa
bash cluster/vllm_stack_4.sh
```

The script starts `vllm serve` in the `vllm` tmux window and exposes it on `127.0.0.1:55154`.

## 6. Useful Links

Add the project-specific links here:

- Artifactory model mirror: `<add_url>`
- Artifactory image repository: `<add_url>`
- Cluster access instructions: `<add_url>`
- SLURM documentation: `<add_url>`
- Run logs location: `/projects/sec_qa/logs`
