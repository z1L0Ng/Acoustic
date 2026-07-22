#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must point to the tracked checkout}"
: "${RESULT_ROOT:?RESULT_ROOT must point to result/add_rsc_YYYYMMDD_HHMMSS}"
: "${ADD_RSC_TRACK:?ADD_RSC_TRACK must name one bounded track}"

case "${ADD_RSC_TRACK}" in
  paper_declared_reconstruction)
    split_protocol="paper_declared_official"
    n_mels="64"
    weight_decay="0.1"
    ;;
  author_repo_default_official_like)
    split_protocol="author_repo_random_file"
    n_mels="128"
    weight_decay="1e-6"
    ;;
  *)
    echo "invalid ADD_RSC_TRACK=${ADD_RSC_TRACK}" >&2
    exit 2
    ;;
esac

resume_args=()
if [[ -n "${ADD_RSC_RESUME:-}" ]]; then
  resume_args=(--resume "${ADD_RSC_RESUME}")
fi

python "${RESULT_ROOT}/source/repo/main.py" \
  --tag "${ADD_RSC_TRACK}_ast_seed0" \
  --dataset icbhi --seed 0 --class_split lungsound --n_cls 4 \
  --epochs 50 --batch_size 8 --optimizer adam \
  --learning_rate 5e-5 --weight_decay "${weight_decay}" \
  --model ast --test_fold official --split_protocol "${split_protocol}" \
  --data_folder "${RESULT_ROOT}/portable_run/data/icbhi_dataset" \
  --sample_rate 16000 --desired_length 8 --pad_types repeat \
  --nfft 1024 --n_mels "${n_mels}" --resz 1 \
  --loss_beta 0.5 --denoise_d_model 256 --denoise_num_heads 8 --denoise_depth 6 \
  --audioset_ckpt "${RESULT_ROOT}/checkpoints/audioset_10_10_0.4593.pth" \
  --save_dir "${RESULT_ROOT}/full/${ADD_RSC_TRACK}" \
  "${resume_args[@]}"
