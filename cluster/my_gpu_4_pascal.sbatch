#!/bin/bash
#SBATCH -p work
#SBATCH -A sec_qa
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --mem 64G
#SBATCH --gres=gpu:rtx1080ti:4
#SBATCH --time=7-00:00:00
#SBATCH -J gpu_4_pascal
#SBATCH -o /projects/sec_qa/logs/%x-%j.out
#SBATCH -e /projects/sec_qa/logs/%x-%j.err

SESSION="inference"

mkdir -p /projects/sec_qa/logs

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n shell
  tmux set-option -t "$SESSION" remain-on-exit on
fi

trap 'exit 0' TERM INT
while true; do
    sleep 300
done
