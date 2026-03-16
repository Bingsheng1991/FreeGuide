#!/bin/bash

set -e

source /home/wbs/FreeGuide/experiments/scripts/server_batch_lib.sh

SEED=42
GPU="$1"

case "$GPU" in
  0)
    run_batch 0 \
      "walker-run|${SEED}|ablation_qev_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
      "humanoid-run|${SEED}|ablation_qev_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
      "walker-run|${SEED}|ablation_edd_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false" \
      "humanoid-run|${SEED}|ablation_edd_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false"
    ;;
  1)
    run_batch 1 \
      "walker-run|${SEED}|ablation_fixed_beta_03_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
      "humanoid-run|${SEED}|ablation_fixed_beta_03_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
      "walker-run|${SEED}|ablation_fixed_beta_01_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1" \
      "humanoid-run|${SEED}|ablation_fixed_beta_01_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1"
    ;;
  2)
    run_batch 2 \
      "walker-run|${SEED}|ablation_fixed_beta_05_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
      "humanoid-run|${SEED}|ablation_fixed_beta_05_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
      "walker-run|${SEED}|ablation_ensemble_K2_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2" \
      "walker-run|${SEED}|ablation_ensemble_K5_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5"
    ;;
  *)
    echo "unknown P2 GPU pipeline: $GPU" >&2
    exit 1
    ;;
esac
