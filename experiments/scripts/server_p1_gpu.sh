#!/bin/bash

set -e

source /home/wbs/FreeGuide/experiments/scripts/server_batch_lib.sh

GPU="$1"

case "$GPU" in
  0)
    run_batch 0 \
      "humanoid-run|1|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false" \
      "humanoid-run|2|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false" \
      "humanoid-run|3|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false" \
      "humanoid-run|4|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 0 \
      "humanoid-run|5|tdmpc2_humanoid-run|freeguide.enabled=false rnd.enabled=false" \
      "humanoid-run|1|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true" \
      "humanoid-run|2|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true" \
      "humanoid-run|3|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 0 \
      "humanoid-run|4|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true" \
      "humanoid-run|5|tdmpc2_rnd_humanoid-run|freeguide.enabled=false rnd.enabled=true" \
      "humanoid-run|1|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false" \
      "humanoid-run|2|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 0 \
      "humanoid-run|3|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false" \
      "humanoid-run|4|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false" \
      "humanoid-run|5|freeguide_humanoid-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 0 \
      "dog-run|1|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false" \
      "dog-run|2|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false" \
      "dog-run|3|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false" \
      "dog-run|4|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 0 \
      "dog-run|5|tdmpc2_dog-run|freeguide.enabled=false rnd.enabled=false" \
      "dog-run|1|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true" \
      "dog-run|2|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true" \
      "dog-run|3|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 0 \
      "dog-run|4|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true" \
      "dog-run|5|tdmpc2_rnd_dog-run|freeguide.enabled=false rnd.enabled=true" \
      "dog-run|1|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false" \
      "dog-run|2|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 0 \
      "dog-run|3|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false" \
      "dog-run|4|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false" \
      "dog-run|5|freeguide_dog-run|freeguide.enabled=true rnd.enabled=false"
    ;;
  1)
    run_batch 1 \
      "quadruped-run|1|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false" \
      "quadruped-run|2|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false" \
      "quadruped-run|3|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false" \
      "quadruped-run|4|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 1 \
      "quadruped-run|5|tdmpc2_quadruped-run|freeguide.enabled=false rnd.enabled=false" \
      "quadruped-run|1|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true" \
      "quadruped-run|2|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true" \
      "quadruped-run|3|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 1 \
      "quadruped-run|4|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true" \
      "quadruped-run|5|tdmpc2_rnd_quadruped-run|freeguide.enabled=false rnd.enabled=true" \
      "quadruped-run|1|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false" \
      "quadruped-run|2|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 1 \
      "quadruped-run|3|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false" \
      "quadruped-run|4|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false" \
      "quadruped-run|5|freeguide_quadruped-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 1 \
      "walker-run|1|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false" \
      "walker-run|2|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false" \
      "walker-run|3|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false" \
      "walker-run|4|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 1 \
      "walker-run|5|tdmpc2_walker-run|freeguide.enabled=false rnd.enabled=false" \
      "walker-run|1|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true" \
      "walker-run|2|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true" \
      "walker-run|3|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 1 \
      "walker-run|4|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true" \
      "walker-run|5|tdmpc2_rnd_walker-run|freeguide.enabled=false rnd.enabled=true" \
      "walker-run|1|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|2|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 1 \
      "walker-run|3|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|4|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false" \
      "walker-run|5|freeguide_walker-run|freeguide.enabled=true rnd.enabled=false"
    ;;
  2)
    run_batch 2 \
      "cheetah-run|1|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false" \
      "cheetah-run|2|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false" \
      "cheetah-run|3|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false" \
      "cheetah-run|4|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false"
    run_batch 2 \
      "cheetah-run|5|tdmpc2_cheetah-run|freeguide.enabled=false rnd.enabled=false" \
      "cheetah-run|1|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true" \
      "cheetah-run|2|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true" \
      "cheetah-run|3|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true"
    run_batch 2 \
      "cheetah-run|4|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true" \
      "cheetah-run|5|tdmpc2_rnd_cheetah-run|freeguide.enabled=false rnd.enabled=true" \
      "cheetah-run|1|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false" \
      "cheetah-run|2|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false"
    run_batch 2 \
      "cheetah-run|3|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false" \
      "cheetah-run|4|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false" \
      "cheetah-run|5|freeguide_cheetah-run|freeguide.enabled=true rnd.enabled=false"
    ;;
  *)
    echo "unknown GPU pipeline: $GPU" >&2
    exit 1
    ;;
esac
