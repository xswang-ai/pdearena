#!/bin/bash
# This script produces 5.2k training, 1.3k valid, and 1.3k test trajectories of the Navier-Stokes dataset.
seeds=(0000)
for SEED in ${seeds[*]};
do
    python3 scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke mode=test samples=1 seed=$SEED pdeconfig.init_args.sample_rate=4 \
    dirname=pdearena_data/navierstokes/
done
