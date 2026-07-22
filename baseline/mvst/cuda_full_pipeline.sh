#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must point to the tracked checkout}"
: "${RESULT_ROOT:?RESULT_ROOT must point to result/mvst_YYYYMMDD_HHMMSS}"

# Author-code fixed random file split. This is not the official ICBHI split.
bash "${PROJECT_ROOT}/baseline/mvst/cuda_encoder_train.sh"

for view in 16 32 64 128 256; do
  checkpoint="${RESULT_ROOT}/full/encoders/${view}/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth"
  for split in train test; do
    python -m baseline.mvst.mvst_extract_features \
      --view "${view}" --split "${split}" \
      --manifest "${RESULT_ROOT}/manifest/icbhi_2017_cycle_manifest.csv" \
      --author-repo "${RESULT_ROOT}/source/repo" \
      --initial-checkpoint "${RESULT_ROOT}/checkpoints/audioset_16_16_0.4422.pth" \
      --task-checkpoint "${checkpoint}" \
      --work-dir "${RESULT_ROOT}/work/extract/${view}" \
      --output "${RESULT_ROOT}/full/features/${view}/${split}.npz" \
      --device cuda
  done
done

resume_args=()
if [[ -n "${MVST_FUSION_RESUME:-}" ]]; then
  resume_args=(--resume "${MVST_FUSION_RESUME}")
fi
python -m baseline.mvst.mvst_fusion_server \
  --feature-root "${RESULT_ROOT}/full/features" \
  --author-repo "${RESULT_ROOT}/source/repo" \
  --output-dir "${RESULT_ROOT}/full/fusion" \
  --device cuda \
  "${resume_args[@]}"

python -m baseline.common.verify_official_predictions \
  --input-npz "${RESULT_ROOT}/full/fusion/official_like_test_outputs.npz" \
  --output-dir "${RESULT_ROOT}/predictions/verified" \
  --expected-rows 2685
