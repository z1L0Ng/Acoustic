#!/usr/bin/env bash
set -euo pipefail

# Minimum compatible snapshot: official-reproduction-release-1.
# Run from the timestamped portable_run directory. The prepared symlink points
# to the current author-hosted runtime AST artifact and its SHA is receipted.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-../work/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-../work/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-../work/cache}"

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--resume "${RESUME_CHECKPOINT}")
fi

python ../source/repo/main.py \
  --tag bs8_lr5e-5_ep50_seed1_best_param \
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
  --method patchmix_cl \
  --temperature 0.06 \
  --proj_dim 768 \
  --alpha 1.0 \
  --mix_beta 1.0 \
  "${resume_args[@]}" \
  --save_dir ../full
