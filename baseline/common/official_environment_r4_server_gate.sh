#!/usr/bin/env bash
set -euo pipefail

# Create new Release 4 environments and fail closed before any bootstrap/training.
PROJECT_ROOT="${1:-$(pwd)}"
cd "$PROJECT_ROOT"

entries=(
  "patch_mix_cl|acoustic-patchmix-r4|baseline/patch_mix_cl/environment.linux-cu118.yml"
  "pafa|acoustic-pafa-r4|baseline/pafa/environment.linux-cu118.yml"
  "sg_scl|acoustic-sgscl-r4|baseline/sg_scl/environment.linux-cu118.yml"
  "mvst|acoustic-mvst-r4|baseline/mvst/environment.linux-cu118.yml"
  "add_rsc|acoustic-addrsc-r4|baseline/add_rsc/environment.linux-cu121.yml"
)

for entry in "${entries[@]}"; do
  IFS='|' read -r method environment spec <<< "$entry"
  if conda env list | awk '{print $1}' | grep -Fxq "$environment"; then
    echo "Refusing to mutate or reuse existing Release 4 environment: $environment" >&2
    exit 1
  fi

  conda env create -f "$spec"
  timestamp="$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
  run_root="result/${method}_${timestamp}"
  mkdir -p "$run_root/receipts"
  conda run -n "$environment" python -m baseline.common.verify_official_environment_r4 \
    --method "$method" \
    --project-root "$PROJECT_ROOT" \
    --cuda-mode runtime \
    --output "$run_root/receipts/environment_r4.json"
  printf '%s\t%s\t%s\n' "$method" "$environment" "$run_root"
done
