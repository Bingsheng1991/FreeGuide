#!/bin/bash

set -e

source /home/wbs/FreeGuide/experiments/scripts/server_batch_lib.sh

run_batch 1 \
  "walker-run|1|ablation_qev_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
  "walker-run|2|ablation_qev_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
  "walker-run|3|ablation_qev_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
  "humanoid-run|1|ablation_qev_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true"
run_batch 0 \
  "humanoid-run|2|ablation_qev_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
  "humanoid-run|3|ablation_qev_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=false freeguide.use_qev=true" \
  "walker-run|1|ablation_edd_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false" \
  "walker-run|2|ablation_edd_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false"
run_batch 1 \
  "walker-run|3|ablation_edd_only_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false" \
  "humanoid-run|1|ablation_edd_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false" \
  "humanoid-run|2|ablation_edd_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false" \
  "humanoid-run|3|ablation_edd_only_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_edd=true freeguide.use_qev=false"
run_batch 2 \
  "walker-run|1|ablation_fixed_beta_03_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
  "walker-run|2|ablation_fixed_beta_03_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
  "walker-run|3|ablation_fixed_beta_03_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
  "humanoid-run|1|ablation_fixed_beta_03_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3"
run_batch 2 \
  "humanoid-run|2|ablation_fixed_beta_03_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3" \
  "humanoid-run|3|ablation_fixed_beta_03_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.3"
run_batch 2 \
  "walker-run|1|ablation_fixed_beta_01_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1" \
  "walker-run|2|ablation_fixed_beta_01_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1" \
  "walker-run|3|ablation_fixed_beta_01_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1" \
  "humanoid-run|1|ablation_fixed_beta_01_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1"
run_batch 2 \
  "humanoid-run|2|ablation_fixed_beta_01_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1" \
  "humanoid-run|3|ablation_fixed_beta_01_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.1"
run_batch 2 \
  "walker-run|1|ablation_fixed_beta_05_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
  "walker-run|2|ablation_fixed_beta_05_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
  "walker-run|3|ablation_fixed_beta_05_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
  "humanoid-run|1|ablation_fixed_beta_05_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5"
run_batch 2 \
  "humanoid-run|2|ablation_fixed_beta_05_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5" \
  "humanoid-run|3|ablation_fixed_beta_05_humanoid-run|freeguide.enabled=true rnd.enabled=false freeguide.use_adaptive_beta=false freeguide.beta_init=0.5"
run_batch 0 \
  "walker-run|1|ablation_ensemble_K2_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2" \
  "walker-run|2|ablation_ensemble_K2_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2" \
  "walker-run|3|ablation_ensemble_K2_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=2" \
  "walker-run|1|ablation_ensemble_K5_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5"
run_batch 1 \
  "walker-run|2|ablation_ensemble_K5_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5" \
  "walker-run|3|ablation_ensemble_K5_walker-run|freeguide.enabled=true rnd.enabled=false freeguide.ensemble_K=5"
