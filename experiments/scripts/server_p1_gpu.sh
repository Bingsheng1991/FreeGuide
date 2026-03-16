#!/bin/bash

set -e

source /home/wbs/FreeGuide/experiments/scripts/server_batch_lib.sh

SEED=42
GPU="$1"

case "$GPU" in
  0)
    run_batch 0 \
      "humanoid-run|${SEED}|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false" \
      "humanoid-run|${SEED}|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true" \
      "humanoid-run|${SEED}|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|${SEED}|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 0 \
      "dog-run|${SEED}|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false"
    ;;
  1)
    run_batch 1 \
      "quadruped-run|${SEED}|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false" \
      "quadruped-run|${SEED}|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true" \
      "quadruped-run|${SEED}|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|${SEED}|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 1 \
      "dog-run|${SEED}|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true"
    ;;
  2)
    run_batch 2 \
      "cheetah-run|${SEED}|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false" \
      "cheetah-run|${SEED}|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true" \
      "cheetah-run|${SEED}|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|${SEED}|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 2 \
      "dog-run|${SEED}|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false"
    ;;
  *)
    echo "unknown GPU pipeline: $GPU" >&2
    exit 1
    ;;
esac
