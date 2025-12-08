#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:${PWD}"

python train/train_vmas.py \
    --env_name="VMAS" \
    --scenario_name="balance" \
    --num_agents=10 \
    --algorithm_name="rmappo" \
    --experiment_name="check" \
    --seed=1 \
    --n_training_threads=1 \
    --n_rollout_threads=1 \
    --num_mini_batch=1 \
    --episode_length=150 \
    --num_env_steps=10050000 \
    --ppo_epoch=10 \
    --use_ReLU \
    --gain=0.01 \
    --lr=7e-4 \
    --critic_lr=7e-4 \
    --wandb_name="DISSCv2" \
    --user_name="hamid-osooli" \
    --save_interval=2000
