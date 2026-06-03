#!/usr/bin/env bash
set -euo pipefail

SESSION="inference"
PORT="55144"

JOB_ID="${SLURM_JOB_ID:?ERROR: run inside Slurm}"

N_GPUS=4
N_CPUS=12

SHELL_CMD="singularity exec --nv --contain --bind /projects/sec_qa:/workspace --pwd /workspace containers/llama9489_avx-cu126-py310-serve.sif"

SRUN_CMD="srun --jobid=$JOB_ID \
  --job-name=llama \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:${N_GPUS} \
  --cpus-per-task=${N_CPUS} \
  --kill-on-bad-exit=1 \
  --export=ALL"

LLAMA_CMD="exec llama-server \
  --host 127.0.0.1 \
  --port $PORT \
  --models-max 1 \
  --models-preset models/models.ini"
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