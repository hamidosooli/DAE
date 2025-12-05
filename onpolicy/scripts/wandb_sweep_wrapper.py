"""
Wandb sweep wrapper for DAE codebase.
Converts wandb config to command-line arguments for training scripts.
"""
import wandb
import subprocess
import sys
import os

def wandb_to_args(config):
    """
    Convert wandb config dictionary to command-line arguments.
    
    Args:
        config: Dictionary from wandb.config
        
    Returns:
        List of command-line argument strings
    """
    args = []
    
    # Special handling for certain parameters
    # Based on config.py analysis:
    # store_true flags (default False, add flag to set True)
    boolean_flags_store_true = {
        "use_eval", "use_render", "save_gifs", "use_linear_lr_decay",
        "use_naive_recurrent_policy", "use_proper_time_limits",
        "use_obs_instead_of_state", "use_stacked_frames", "use_popart"
    }
    
    # store_false flags (default True, add flag to set False)
    # Note: use_wandb is store_false with default=False (confusing, but that's how it is)
    boolean_flags_store_false = {
        "cuda", "cuda_deterministic", "use_ReLU", "use_valuenorm",
        "use_feature_normalization", "use_orthogonal", "use_recurrent_policy",
        "use_centralized_V", "share_policy", "use_clipped_value_loss",
        "use_max_grad_norm", "use_gae", "use_huber_loss", "use_value_active_masks",
        "use_policy_active_masks"
    }
    
    # Special case: use_wandb is store_false with default=False
    # For wandb sweeps, we want wandb enabled, so we don't pass the flag
    # (since passing it would set it to False)
    
    for key, value in config.items():
        # Skip special wandb keys
        if key in ["_wandb", "program", "method", "metric"]:
            continue
        
        # Handle nested dictionaries (e.g., env_args.time_limit)
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                arg_key = f"--{key}_{subkey}" if key != "env_args" else f"--{subkey}"
                if isinstance(subvalue, bool):
                    if subvalue and arg_key.replace("--", "") in boolean_flags_store_true:
                        args.append(arg_key)
                    elif not subvalue and arg_key.replace("--", "") in boolean_flags_store_false:
                        args.append(arg_key)
                    elif not isinstance(subvalue, bool):
                        args.append(f"{arg_key}={subvalue}")
                else:
                    args.append(f"{arg_key}={subvalue}")
        # Handle boolean flags
        elif isinstance(value, bool):
            # Special case: use_wandb is store_false with default=False
            # For wandb sweeps, we always want wandb enabled, so skip this parameter
            if key == "use_wandb":
                continue
            elif key in boolean_flags_store_true:
                # store_true: add flag only if True
                if value:
                    args.append(f"--{key}")
            elif key in boolean_flags_store_false:
                # store_false: add flag only if False (to disable default True)
                if not value:
                    args.append(f"--{key}")
            else:
                # Unknown boolean flag - treat as store_true
                if value:
                    args.append(f"--{key}")
        # Handle None values (skip them)
        elif value is None:
            continue
        # Handle lists/arrays - wandb sweeps typically provide single values
        elif isinstance(value, (list, tuple)):
            # For most cases in sweeps, we get a single value, but handle lists
            if len(value) == 1:
                args.append(f"--{key}={value[0]}")
            elif len(value) > 1:
                # If multiple values, join them (though this is rare for argparse)
                args.append(f"--{key}={','.join(map(str, value))}")
        # Handle regular values (int, float, str)
        else:
            args.append(f"--{key}={value}")
    
    return args

def get_training_script(env_name):
    """Get the appropriate training script based on environment name."""
    script_map = {
        "MPE": "onpolicy/scripts/train/train_mpe.py",
        "VMAS": "onpolicy/scripts/train/train_vmas.py",
        "StarCraft2": "onpolicy/scripts/train/train_smac.py",
        "Hanabi": "onpolicy/scripts/train/train_hanabi_forward.py",
    }
    return script_map.get(env_name, "onpolicy/scripts/train/train_mpe.py")

if __name__ == "__main__":
    # Initialize wandb (this will load the sweep config)
    # For sweeps, wandb is initialized by the sweep agent, so we use wandb.init() to join the run
    wandb.init()
    
    # Get the config from wandb
    config = wandb.config
    
    # Determine which training script to use
    env_name = config.get("env_name", "MPE")
    training_script = get_training_script(env_name)
    
    # Convert wandb config to command-line arguments
    args = wandb_to_args(config)
    
    # Note: use_wandb is store_false with default=False in config.py
    # This means we can't set it to True via command line
    # However, the training script will check use_wandb and call wandb.init() with reinit=True
    # Since we're already in a wandb run, the training script's wandb.init() will join/reinit properly
    # We need to ensure use_wandb is True, but since we can't set it via CLI, we'll need to
    # modify the approach. For now, we'll let the training script handle wandb initialization
    # with reinit=True, which should work even if use_wandb is False (it will just use tensorboard).
    # Actually, to enable wandb in the training script, we need use_wandb=True, but we can't set it.
    # The workaround: set an environment variable that the training script can check
    
    # Set environment variable to indicate we're in a wandb sweep
    os.environ["WANDB_SWEEP"] = "1"
    
    # Build the command
    command = ["python", training_script] + args
    
    print("[DEBUG] Launching:", " ".join(command))
    print(f"[DEBUG] Working directory: {os.getcwd()}")
    
    # Execute the command
    result = subprocess.call(command)
    
    # Exit with the same code as the subprocess
    sys.exit(result)

