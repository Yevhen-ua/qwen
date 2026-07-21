#!/usr/bin/env bash
set -euo pipefail

SESSION="inference"
PORT="55144"
ACTION="${1:-start}"

JOB_ID="${SLURM_JOB_ID:?ERROR: run inside Slurm}"

N_GPUS=4
N_CPUS=12

SHELL_CMD="singularity exec --nv --contain --bind /projects/sec_qa:/workspace --pwd /workspace containers/llama9436_avx-cu126-py310-serve.sif"

SRUN_CMD="srun --jobid=$JOB_ID \
  --job-name=llama \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:${N_GPUS} \
  --cpus-per-task=${N_CPUS} \
  --kill-on-bad-exit=1 \
  --quit-on-interrupt \
  --export=ALL"

LLAMA_CMD="exec llama-server \
  -m models/unsloth-Qwen36-27B-MTP-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --mmproj models/unsloth-Qwen36-27B-MTP-GGUF/mmproj-F16.gguf \
  --no-ui \
  --threads-http 2 \
  --threads ${N_CPUS} \
  --threads-batch ${N_CPUS} \
  --fit off \
  --flash-attn on \
  --jinja \
  --parallel 1 \
  --ctx-size 67584 \
  --n-gpu-layers all \
  --no-mmap \
  --mlock \
  --split-mode layer \
  --tensor-split 1,1,1,1 \
  --host 127.0.0.1 \
  --port $PORT \
  --alias Qwen3.6-27B-UD-Q5_K_X \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning off \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --cache-type-k-draft q8_0 \
  --cache-type-v-draft q8_0 \
  --temp 0.7 \
  --top-p 0.80 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0"
TUNNEL_CMD="ssh -N -R $PORT:127.0.0.1:$PORT some_user@106.125.46.169"

has_window() {
  tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"
}

find_llama_step() {
  squeue -h -s -j "$JOB_ID" -o "%i %j" | awk '$2 == "llama" { print $1; exit }'
}

stop_llama() {
  local step_id
  step_id="$(find_llama_step)"

  if [[ -z "${step_id:-}" ]]; then
    echo "llama step not found for job $JOB_ID" >&2
    return 1
  fi

  echo "Stopping step $step_id"
  scancel "$step_id"
}

kill_llama() {
  local step_id
  step_id="$(find_llama_step)"

  if [[ -z "${step_id:-}" ]]; then
    echo "llama step not found for job $JOB_ID" >&2
    return 1
  fi

  echo "Killing step $step_id"
  scancel --signal=KILL "$step_id"
}

case "$ACTION" in
  start)
    ;;
  stop-llama)
    stop_llama
    exit 0
    ;;
  kill-llama)
    kill_llama
    exit 0
    ;;
  *)
    echo "Usage: $0 [start|stop-llama|kill-llama]" >&2
    exit 2
    ;;
esac

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
  tmux set-option -t "$SESSION" remain-on-exit on
fi

if ! has_window "shell"; then
  tmux new-window -t "$SESSION" -n shell
  tmux send-keys -t "$SESSION:shell" "pwd" C-m
fi

if ! has_window "llama"; then
  tmux new-window -t "$SESSION" -n llama

  LLAMA_CMD_ESCAPED=$(printf '%q' "$LLAMA_CMD")

  tmux send-keys -t "$SESSION:llama" "$SRUN_CMD $SHELL_CMD bash -lc $LLAMA_CMD_ESCAPED" C-m
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
