#!/bin/bash

# Suppress warnings at the environment level
export PYTHONWARNINGS="ignore"
ENV="walker_walk"
STEPS=1
SEEDS=(1)

for SEED in "${SEEDS[@]}"
do
    echo "------------------------------"
    echo "RUNNING SEED: $SEED"
    echo "------------------------------"

    # SAC Oracle
    python3 -W ignore train_SAC.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        hydra.run.dir=exp_local/oracle_sac_s${SEED}

    # PPO Oracle
    python3 -W ignore train_PPO.py --env $ENV --seed $SEED --total-timesteps $STEPS \
        --tensorboard-log exp_local/oracle_ppo_s${SEED}

    # PrefPPO Baselines
    python3 -W ignore train_PrefPPO.py --env $ENV --seed $SEED --total-timesteps $STEPS \
        --re-max-feed 2100 --unsuper-step 0 \
        --tensorboard-log exp_local/prefppo_2100_s${SEED}

    python3 -W ignore train_PrefPPO.py --env $ENV --seed $SEED --total-timesteps $STEPS \
        --re-max-feed 1400 --unsuper-step 0 \
        --tensorboard-log exp_local/prefppo_1400_s${SEED}

    python3 -W ignore train_PrefPPO.py --env $ENV --seed $SEED --total-timesteps $STEPS \
        --re-max-feed 1400 --unsuper-step 5000 \
        --tensorboard-log exp_local/prefppo_pretrain_1400_s${SEED}

    # PEBBLE Baselines
    python3 -W ignore train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=1400 num_unsup_steps=5000 \
        hydra.run.dir=exp_local/pebble_1400_s${SEED}

    python3 -W ignore train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=700 num_unsup_steps=5000 \
        hydra.run.dir=exp_local/pebble_700_s${SEED}

    python3 -W ignore train_PEBBLE.py env=$ENV seed=$SEED num_train_steps=$STEPS \
        max_feedback=400 num_unsup_steps=5000 \
        hydra.run.dir=exp_local/pebble_400_s${SEED}
done
