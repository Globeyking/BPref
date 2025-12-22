#!/bin/bash

# Configuration
ENV="walker_walk"
STEPS=50000 
# We will iterate through seeds 1, 2, and 3
SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"
do
    echo "=============================="
    echo "RUNNING SEED: $SEED"
    echo "=============================="

    # 1. ORACLE
    python train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=0 num_unsup_steps=0 \
        experiment=oracle_s${SEED}

    # 2. PEBBLE
    python train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=500 num_unsup_steps=5000 \
        experiment=pebble_s${SEED}

    # 3. SAC + UNSUP
    python train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=0 num_unsup_steps=5000 \
        experiment=sac_s${SEED}
done

echo "All 9 runs complete (3 algorithms x 3 seeds)."
