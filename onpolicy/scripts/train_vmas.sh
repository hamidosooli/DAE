#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:${PWD}"

python onpolicy/scripts/train/train_vmas.py \
    --env_name="VMAS" \
    --scenario_name="navigation" \
    --num_agents=2 \
    --algorithm_name="rmappo" \
    --experiment_name="check" \
    --seed=1 \
    --n_training_threads=1 \
    --n_rollout_threads=1 \
    --num_mini_batch=1 \
    --episode_length=100 \
    --num_env_steps=10000000 \
    --ppo_epoch=10 \
    --use_ReLU \
    --gain=0.01 \
    --lr=7e-4 \
    --critic_lr=7e-4 \
    --wandb_name="mappo_vmas" \
    --user_name="mappo"

