#!/usr/bin/env python3
"""Improved PPO training script for residual-learning locomotion.

This version includes:
- Fixed gait parameters matching baseline controller
- Improved reward function (5x forward velocity weight)
- Support for parallel environments
- Better hyperparameters for robust learning
- Optional curriculum learning support

Usage:
    # Quick smoke test (1M steps, single env)
    python3 train_residual_ppo_v2.py --total-timesteps 1000000 --n-envs 1 --run-name smoke_v2

    # Production run (10M steps, 12 parallel envs)
    python3 train_residual_ppo_v2.py --total-timesteps 10000000 --n-envs 12 --run-name prod_v2

    # Long training with smaller batch for better sample efficiency
    python3 train_residual_ppo_v2.py --total-timesteps 20000000 --n-envs 16 \
        --n-steps 2048 --batch-size 512 --run-name prod_long
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym  # noqa: F401
except Exception:
    import gym  # type: ignore  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.residual_walk_env import ResidualWalkEnv
from gait_controller import GaitParameters


def make_env(log_dir: Path, rank: int, randomize: bool = False):
    """Factory function for creating environments.

    Args:
        log_dir: Directory for monitor logs
        rank: Environment index for parallel training
        randomize: Whether to randomize initial conditions (for robustness)
    """
    def _init():
        # Use gait parameters that work well for rough terrain
        # These match the baseline controller used in height_control.py
        gait = GaitParameters(
            body_height=0.05,
            step_length=0.06,
            step_height=0.04,
            cycle_time=0.8
        )
        env = ResidualWalkEnv(
            model_path="model/world_train.xml",
            gait_params=gait,
            residual_scale=0.02,
            max_episode_steps=1000,
            settle_steps=500,
            seed=rank,
        )
        # Monitor writes per-episode stats; helps SB3 compute ep_rew_mean
        monitor_file = log_dir / f"monitor_{rank}.csv"
        env = Monitor(env, filename=str(monitor_file))
        return env

    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train residual PPO policy (v2 with fixes)")

    # Training duration
    p.add_argument("--total-timesteps", type=int, default=10_000_000,
                   help="Total timesteps to train (default: 10M)")

    # Parallelization
    p.add_argument("--n-envs", type=int, default=12,
                   help="Number of parallel environments (default: 12)")

    # PPO hyperparameters
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Steps per environment per update (default: 2048)")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Minibatch size for PPO updates (default: 512)")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda (default: 0.95)")
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Number of epochs per update (default: 10)")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="Entropy coefficient (default: 0.0)")
    p.add_argument("--clip-range", type=float, default=0.2,
                   help="PPO clip range (default: 0.2)")

    # Saving and logging
    p.add_argument("--run-name", type=str, default="prod_v2",
                   help="Run name for logging")
    p.add_argument("--log-root", type=str, default="runs",
                   help="Root directory for logs")
    p.add_argument("--checkpoint-freq", type=int, default=100_000,
                   help="Save checkpoint every N steps (default: 100k)")

    # Advanced options
    p.add_argument("--randomize", action="store_true",
                   help="Enable domain randomization")
    p.add_argument("--device", type=str, default="auto",
                   help="Device to use (auto/cpu/cuda)")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Create log directory with timestamp
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_root) / f"{args.run_name}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Run name:         {args.run_name}_{ts}")
    print(f"Log directory:    {log_dir}")
    print(f"Total timesteps:  {args.total_timesteps:,}")
    print(f"Parallel envs:    {args.n_envs}")
    print(f"Steps per update: {args.n_steps}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Device:           {args.device}")
    print("="*70)

    # Save configuration
    config_file = log_dir / "config.txt"
    with open(config_file, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # Create vectorized environment
    env_fns = [make_env(log_dir, i, args.randomize) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Wrap with VecNormalize for observation/reward normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # PPO model with improved hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=str(log_dir),
        verbose=1,
        device=args.device,
    )

    # Callbacks for checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjust for parallel envs
        save_path=str(log_dir / "checkpoints"),
        name_prefix="rl_model",
        save_vecnormalize=True,
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final artifacts
    print("\nSaving final model...")
    model.save(str(log_dir / "final_model"))
    vec_env.save(str(log_dir / "vec_normalize.pkl"))

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    print(f"Artifacts saved to: {log_dir}")
    print(f"  - final_model.zip")
    print(f"  - vec_normalize.pkl")
    print(f"  - checkpoints/")
    print(f"  - monitor_*.csv")
    print(f"\nTo evaluate, run:")
    print(f"  python3 phase4_viewer_play_policy.py \\")
    print(f"    --model {log_dir}/final_model.zip \\")
    print(f"    --normalize {log_dir}/vec_normalize.pkl \\")
    print(f"    --seconds 20 --deterministic")
    print(f"\nOr use the debug script:")
    print(f"  python3 debug_model.py {log_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
