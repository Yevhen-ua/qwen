#!/usr/bin/env bash
set -euo pipefail

SESSION="inference"
PORT="55154"

JOB_ID="${SLURM_JOB_ID:?ERROR: run inside Slurm}"

N_GPUS=4
N_CPUS=12

SHELL_CMD="singularity exec --nv --contain --bind /projects/sec_qa:/workspace --pwd /workspace containers/vllm0230-cu131-py312-serve.sif"

SRUN_CMD="srun --jobid=$JOB_ID \
  --job-name=vllm \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:${N_GPUS} \
  --cpus-per-task=${N_CPUS} \
  --kill-on-bad-exit=1 \
  --export=ALL"

VLLM_CMD="mkdir -p /dev/shm/vllm-rpc && VLLM_USE_V2_MODEL_RUNNER=1 CUDA_SCALE_LAUNCH_QUEUES=4x FLASHINFER_DISABLE_VERSION_CHECK=1 exec vllm serve models/cyankiwi-Qwen36-27B-AWQ-INT4 \
  --served-model-name Qwen36-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port $PORT \
  --mm-processor-cache-type shm \
  --tensor-parallel-size $N_GPUS \
  --dtype auto \
  --gpu-memory-utilization 0.975 \
  --max-model-len 38912 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --limit-mm-per-prompt.video 0 \
  --limit-mm-per-prompt.image 1 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{\"enable_thinking\":false}' \
  --max-num-batched-tokens 4096"
TUNNEL_CMD="ssh -N -R $PORT:127.0.0.1:$PORT some_user@106.125.46.169"

has_window() {
  tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"
}

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
  tmux set-option -t "$SESSION" remain-on-exit on
fi

if ! has_window "shell"; then
  tmux new-window -t "$SESSION" -n shell
  tmux send-keys -t "$SESSION:shell" "pwd" C-m
fi

if ! has_window "vllm"; then
  tmux new-window -t "$SESSION" -n vllm

  VLLM_CMD_ESCAPED=$(printf '%q' "$VLLM_CMD")

  tmux send-keys -t "$SESSION:vllm" "$SRUN_CMD $SHELL_CMD bash -lc $VLLM_CMD_ESCAPED" C-m
fi

if ! has_window "tunnel"; then
  tmux new-window -t "$SESSION" -n tunnel
  tmux send-keys -t "$SESSION:tunnel" "
while true; do
  echo \"[\$(date '+%F %T')] starting tunnel\"
  $TUNNEL_CMD
  code=\$?
  echo \"[\$(date '+%F %T')] tunnel exited with code \$code, restarting in 10s\"
  sleep 10
done
" C-m
fi

exec tmux attach -t "$SESSION"
