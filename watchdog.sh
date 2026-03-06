#!/bin/bash
# Auto-restart watchdog: survives carb mutex assertion crashes in Isaac Sim

BASE=/root/autodl-tmp/thunder2/ame2_standalone
PYTHON=/root/autodl-tmp/conda_envs/thunder2/bin/python
LOG=$BASE/train_direct_v4_gpu0.log
CRASH_LOG=$BASE/watchdog_crashes.log

run=0
while true; do
  run=$((run + 1))
  # sort -V: version sort picks highest-numbered checkpoint (e.g. model_75.pt over model_0.pt)
  # ls -t sorts by mtime which is wrong: model_0.pt is overwritten on every run start
  CKPT=$(ls $BASE/logs_direct_v4/model_*.pt 2>/dev/null | sort -V | tail -1)
  echo "[$(date)] Run $run started. Checkpoint: ${CKPT:-none}" >> $CRASH_LOG

  if [ -n "$CKPT" ]; then
    cd $BASE && CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PYTHON -u scripts/train_ame2_direct.py \
      --num_envs 2048 --max_iterations 80000 --seed 42 \
      --log_dir logs_direct_v4 --headless \
      --resume "$CKPT" >> $LOG 2>&1
  else
    cd $BASE && CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PYTHON -u scripts/train_ame2_direct.py \
      --num_envs 2048 --max_iterations 80000 --seed 42 \
      --log_dir logs_direct_v4 --headless >> $LOG 2>&1
  fi

  EXIT=$?
  echo "[$(date)] Run $run exited with code $EXIT" >> $CRASH_LOG

  if [ $EXIT -eq 0 ] || [ -f $BASE/STOP_TRAINING ]; then
    echo "[$(date)] Done or stop requested." >> $CRASH_LOG
    break
  fi

  echo "[$(date)] Crash detected (exit=$EXIT), restarting in 15s..." >> $CRASH_LOG
  sleep 15
done
