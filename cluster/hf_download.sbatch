#!/usr/bin/env bash
#SBATCH -p work
#SBATCH -A sec_qa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem 8G
#SBATCH --time=1-00:00:00
#SBATCH -J hf_download
#SBATCH -o /projects/sec_qa/logs/%x-%j.out
#SBATCH -e /projects/sec_qa/logs/%x-%j.err

# Examples:
#   HF_TOKEN=... MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct sbatch cluster/hf_download.sbatch
#   HF_TOKEN=... MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct HF_FILE=config.json sbatch cluster/hf_download.sbatch
#   HF_TOKEN=... MODEL_ID=repo/name HF_FILES="model.gguf mmproj.gguf" sbatch cluster/hf_download.sbatch
#   HF_TOKEN=... MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct LOCAL_DIR=/workspace/models/qwen-vl sbatch cluster/hf_download.sbatch
#
# Defaults:
#   CONTAINER=containers/huggingface-hub-py312.sif
#   WORKSPACE=/projects/sec_qa
#   LOCAL_DIR=/workspace/models/<MODEL_ID with "/" replaced by "__">

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before sbatch, e.g. HF_TOKEN=... sbatch cluster/hf_download.sbatch}"
: "${MODEL_ID:?Set MODEL_ID before sbatch, e.g. MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct}"

CONTAINER="${CONTAINER:-containers/huggingface-hub-py312.sif}"
WORKSPACE="${WORKSPACE:-/projects/sec_qa}"
LOCAL_DIR="${LOCAL_DIR:-/workspace/models/${MODEL_ID//\//__}}"

mkdir -p "${WORKSPACE}/logs"

HF_DOWNLOAD_ARGS=("${MODEL_ID}" "--local-dir" "${LOCAL_DIR}")
if [ -n "${HF_FILE:-}" ]; then
    HF_DOWNLOAD_ARGS+=("${HF_FILE}")
fi
if [ -n "${HF_FILES:-}" ]; then
    read -r -a HF_FILE_LIST <<< "${HF_FILES}"
    HF_DOWNLOAD_ARGS+=("${HF_FILE_LIST[@]}")
fi

singularity exec --containall \
    --env HF_TOKEN="${HF_TOKEN}" \
    --bind "${WORKSPACE}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    hf download "${HF_DOWNLOAD_ARGS[@]}"
