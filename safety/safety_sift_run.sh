#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Grid search launcher: DPO + SFT
# - lr:    9e-4 5e-4 5e-5
# - beta:  0.1  0.05 0.01
# - gamma: 0.1  0.5
# Run 4 jobs concurrently, then wait, then next 4.
# Logs and outputs go into per-run OUTPUT_DIR.
# ============================================================

# ---- Fixed defaults (edit if needed) ------------------------
MODEL="meta-llama/Llama-2-7b-hf"
BASE_OUTPUT_DIR="./muon_v1"
EPOCHS=3
BATCH=4
GRAD_ACC=8
ALPACA_RATIO=1.0
MAX_LEN=512
MAX_PROMPT_LEN=512
SEED=0
PREFER_SAFE="True"
REPORT_TO="none"   # change to wandb if you want
MAX_CONCURRENT=4

# Pass-through extra flags to python (e.g. --sanity_check True)
EXTRA_ARGS="${@:-}"

# ---- Grid ---------------------------------------------------
LRS=("3e-5" "1e-5" "5e-5" "7e-5" "1e-4" "5e-4")
BETAS=("0.01")
GAMMAS=("1")

# ---- Helpers ------------------------------------------------
sanitize() {
  # make a string safe for folder names
  # e.g., 9e-4 -> 9e-4, 0.05 -> 0p05
  echo "$1" | sed 's/\./p/g'
}

run_one() {
  local lr="$1"
  local beta="$2"
  local gamma="$3"

  local lr_tag beta_tag gamma_tag out_dir log_file
  lr_tag="$(sanitize "$lr")"
  beta_tag="$(sanitize "$beta")"
  gamma_tag="$(sanitize "$gamma")"

  out_dir="${BASE_OUTPUT_DIR}_lr${lr_tag}_beta${beta_tag}_gamma${gamma_tag}"
  mkdir -p "${out_dir}"
  log_file="${out_dir}/train.log"

  echo "[LAUNCH] lr=${lr} beta=${beta} gamma=${gamma} -> ${out_dir}"

  # run in background; redirect stdout/stderr to log
  python dpo_sft_train_muon_v1.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$out_dir" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --learning_rate "$lr" \
    --beta "$beta" \
    --gamma "$gamma" \
    --sft_alpaca_ratio "$ALPACA_RATIO" \
    --max_length "$MAX_LEN" \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --prefer_safe "$PREFER_SAFE" \
    --seed "$SEED" \
    --report_to "$REPORT_TO" \
    $EXTRA_ARGS \
    >"$log_file" 2>&1 & 
}

# ---- Main loop ----------------------------------------------
pids=()
count_in_batch=0
total=0

# for beta in "${BETAS[@]}"; do
# for lr in "${LRS[@]}"; do
# for gamma in "${GAMMAS[@]}"; do

for gamma in "${GAMMAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for lr in "${LRS[@]}"; do
      run_one "$lr" "$beta" "$gamma"
      pids+=("$!")
      count_in_batch=$((count_in_batch + 1))
      total=$((total + 1))

      # If batch full, wait them all
      if [[ "$count_in_batch" -ge "$MAX_CONCURRENT" ]]; then
        echo "===== Waiting for batch of ${count_in_batch} jobs... ====="
        for pid in "${pids[@]}"; do
          wait "$pid"
        done
        echo "===== Batch finished. ====="
        pids=()
        count_in_batch=0
      fi
    done
  done
done

# wait leftover jobs
if [[ "$count_in_batch" -gt 0 ]]; then
  echo "===== Waiting for final batch of ${count_in_batch} jobs... ====="
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "===== Final batch finished. ====="
fi

echo "ALL DONE. Total runs: ${total}"