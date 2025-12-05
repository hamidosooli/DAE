# Wandb Sweeps for DAE Codebase

This guide explains how to use Wandb sweeps for hyperparameter tuning with the DAE codebase.

## Setup

1. Make sure you have wandb installed:
```bash
pip install wandb
```

2. Login to wandb (if not already):
```bash
wandb login
```

## Configuration Files

The sweep configuration is defined in `search.config.wandb.rmappo.yaml`. This file specifies:
- The program to run (`wandb_sweep_wrapper.py`)
- The search method (random, grid, bayes)
- The metric to optimize
- The hyperparameters to tune and their search spaces

## Running a Sweep

### Initialize a Sweep

From the DAE root directory:

```bash
wandb sweep search.config.wandb.rmappo.yaml
```

This will output a sweep ID like `your-entity/your-project/sweep-id`.

### Run Sweep Agents

To run agents that will execute the sweep trials:

```bash
wandb agent your-entity/your-project/sweep-id
```

You can run multiple agents in parallel (on different machines/GPUs) to speed up the sweep.

### Example

```bash
# Initialize sweep
wandb sweep search.config.wandb.rmappo.yaml

# Output: your-entity/DAE/abc123xyz

# Run agent (can run multiple in parallel)
wandb agent your-entity/DAE/abc123xyz
```

## Customizing the Sweep

Edit `search.config.wandb.rmappo.yaml` to:
- Change the search method (`random`, `grid`, `bayes`)
- Modify hyperparameter search spaces
- Change the metric to optimize
- Add/remove parameters to tune

### Parameter Types in YAML

- `value: X` - Fixed value (not tuned)
- `values: [X, Y, Z]` - Categorical choices
- `min: X, max: Y` - Continuous range (for bayes/grid search)
- `distribution: uniform, min: X, max: Y` - Distribution for random search

## How It Works

1. `wandb_sweep_wrapper.py` is called by wandb with a config
2. The wrapper converts the wandb config to command-line arguments
3. It calls the appropriate training script (based on `env_name`)
4. The training script runs with the hyperparameters from the sweep

## Supported Environments

The wrapper automatically selects the correct training script based on `env_name`:
- `MPE` → `train_mpe.py`
- `VMAS` → `train_vmas.py`
- `StarCraft2` → `train_smac.py`
- `Hanabi` → `train_hanabi_forward.py`

## Troubleshooting

- **Boolean flags**: The wrapper handles `store_true` and `store_false` flags correctly based on the config.py definitions
- **Nested parameters**: Parameters like `env_args.time_limit` are converted to `--time_limit`
- **use_wandb**: This is handled specially - wandb is always enabled for sweeps

## Viewing Results

Results are automatically logged to wandb. View them at:
- https://wandb.ai/your-entity/your-project

You can compare runs, visualize metrics, and analyze hyperparameter importance.

