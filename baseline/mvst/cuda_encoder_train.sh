#!/usr/bin/env bash
set -euo pipefail

# Run from result/mvst_<timestamp>/portable_run on CUDA Linux. This preserves
# the author's fixed random file split and test-selected encoder checkpoints.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-../work/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-../work/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-../work/cache}"

for view in 16 32 64 128 256; do
  resume_var="MVST_RESUME_${view}"
  resume_args=()
  if [[ -n "${!resume_var:-}" ]]; then
    resume_args=(--resume "${!resume_var}")
  fi
  python "${RESULT_ROOT}/source/repo/${view}/main.py" \
    --tag bs8_lr5e-5_ep50_seed1 \
    --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
    --epochs 50 --batch_size 8 --optimizer adam \
    --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
    --model ast --test_fold official --pad_types repeat --resz 1 \
    --n_mels 128 --ma_update --ma_beta 0.5 \
    --from_sl_official --audioset_pretrained --method ce \
    --save_dir "${RESULT_ROOT}/full/encoders/${view}" \
    "${resume_args[@]}"
done
