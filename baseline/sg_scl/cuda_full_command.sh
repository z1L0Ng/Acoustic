#!/usr/bin/env bash
set -euo pipefail

# Run from the timestamped portable_run directory on a CUDA Linux host.
# The author evaluates the official test each epoch and selects by test Score.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-../work/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-../work/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-../work/cache}"

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--resume "${RESUME_CHECKPOINT}")
fi

python ../source/repo/main.py \
  --tag sg_scl_bs8_lr5e-5_ep50_seed1_best_param \
  --dataset icbhi \
  --seed 1 \
  --class_split lungsound \
  --n_cls 4 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --model ast \
  --test_fold official \
  --pad_types repeat \
  --resz 1 \
  --n_mels 128 \
  --ma_update \
  --ma_beta 0.5 \
  --from_sl_official \
  --audioset_pretrained \
  --method ce \
  --domain_adaptation2 \
  --alpha 1.0 \
  --proj_dim 768 \
  --temperature 0.06 \
  --meta_mode dev \
  --target_type project1_project2block \
  --save_freq 1 \
  --save_dir ../full \
  "${resume_args[@]}"
