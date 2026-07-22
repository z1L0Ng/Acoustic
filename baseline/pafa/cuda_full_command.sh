#!/usr/bin/env bash
set -euo pipefail

# Minimum compatible snapshot: Release 3 based on 3f757adcc12f.
# Run from result/pafa_<timestamp>/portable_run on a CUDA Linux host. The
# current public compatibility checkpoint keeps this run official-like until
# Microsoft-hosted byte identity can be independently confirmed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-../work/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-../work/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-../work/cache}"

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--resume "${RESUME_CHECKPOINT}")
fi

python ../source/repo/main.py \
  --tag seed1_best \
  --dataset icbhi \
  --seed 1 \
  --class_split lungsound \
  --n_cls 4 \
  --epochs 100 \
  --batch_size 32 \
  --desired_length 5 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --model beats \
  --test_fold official \
  --pad_types repeat \
  --resz 1 \
  --n_mels 128 \
  --ma_update \
  --ma_beta 0.5 \
  --from_sl_official \
  --audioset_pretrained \
  --method pafa \
  --w_ce 1.0 \
  --w_pafa 1.0 \
  --lambda_pcsl 50.0 \
  --lambda_gpal 0.0005 \
  --norm_type ln \
  --output_dim 768 \
  --nospec \
  "${resume_args[@]}" \
  --save_dir ../full
