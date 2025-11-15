#!/usr/bin/env python3
"""PPO training script for adaptive gait controller.

This script trains a policy that learns to modulate gait parameters
(step_height, step_length, cycle_time, body_height) in addition to
residual corrections. This approach enables better adaptation to rough terrain.

Key differences from residual-only training:
- Action space: 16D (4 param deltas + 12 residuals) vs 12D (residuals only)
- Observation: 69D (includes current gait params) vs 65D
- Policy learns both high-level (gait params) and low-level (residuals) control

Usage:
    Run this script directly. To change training settings, edit the
    TrainingConfig dataclass below.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym  # noqa: F401
except Exception:
    import gym  # type: ignore  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.adaptive_gait_env import AdaptiveGaitEnv
from gait_controller import GaitParameters

RESIDUAL_SCALE = 0.01
@dataclass
class TrainingConfig:
    """Centralized training configuration."""
    # Duration
    total_timesteps: int = 30_000_000  # Increased for more complex policy

    # Parallelism
    n_envs: int = 84
    vec_env_backend: str = "subproc"  # "subproc" or "dummy"

    # PPO hyperparameters
    n_steps: int = 4096
    batch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 10
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 1.0

    # Network and env
    network_size: str = "large"  # small | medium | large
    residual_scale: float = RESIDUAL_SCALE

    # Logging/saving
    run_name: str = "adaptive_gait"
    log_root: str = "runs"
    checkpoint_freq: int = 500_000

    # Misc
    randomize: bool = False
    device: str = "auto"  # auto | cpu | cuda
    seed: int | None = None


def make_env(log_dir: Path, rank: int, cfg: TrainingConfig):
    """Factory function for creating environments."""
    def _init():
        # Base gait parameters (policy will learn to adapt these)
        gait = GaitParameters(
            body_height=0.05,
            step_length=0.06,
            step_height=0.04,
            cycle_time=0.8
        )
        env = AdaptiveGaitEnv(
            model_path="model/world_train.xml",
            gait_params=gait,
            residual_scale=cfg.residual_scale,
            max_episode_steps=60000,
            settle_steps=0,
            seed=rank,
        )
        monitor_file = log_dir / f"monitor_{rank}.csv"
        env = Monitor(env, filename=str(monitor_file))
        return env

    return _init


def get_network_architecture(size: str) -> dict:
    """Get network architecture based on size parameter."""
    architectures = {
        "small": [128, 64],
        "medium": [256, 128, 64],
        "large": [512, 256, 128],
    }

    net_arch = architectures[size]

    policy_kwargs = dict(
        net_arch=dict(
            pi=net_arch,  # Actor (policy)
            vf=net_arch,  # Critic (value function)
        ),
        activation_fn=nn.ELU,
    )

    return policy_kwargs


def main() -> int:
    cfg = TrainingConfig()

    # Set random seed if provided
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Create log directory with timestamp
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.log_root) / f"{cfg.run_name}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 80)
    print("Adaptive Gait Training Configuration")
    print("=" * 80)
    print(f"Run name:            {cfg.run_name}_{ts}")
    print(f"Log directory:       {log_dir}")
    print(f"Total timesteps:     {cfg.total_timesteps:,}")
    print(f"Parallel envs:       {cfg.n_envs}")
    print(f"Steps per update:    {cfg.n_steps}")
    print(f"Batch size:          {cfg.batch_size}")
    print(f"Learning rate:       {cfg.learning_rate}")
    print(f"Entropy coef:        {cfg.ent_coef}")
    print(f"N epochs:            {cfg.n_epochs}")
    print(f"Max grad norm:       {cfg.max_grad_norm}")
    print(f"Network size:        {cfg.network_size}")
    print(f"Residual scale:      {cfg.residual_scale}")
    print(f"Device:              {cfg.device}")
    backend = "SubprocVecEnv" if (cfg.vec_env_backend == "subproc" and cfg.n_envs > 1) else "DummyVecEnv"
    print(f"Vec env backend:     {backend}")
    if cfg.seed is not None:
        print(f"Random seed:         {cfg.seed}")
    print("=" * 80)
    print()
    print("Key differences from residual-only training:")
    print("  - Action space: 16D (4 param deltas + 12 residuals)")
    print("  - Observation: 69D (includes current gait params)")
    print("  - Policy learns both high-level and low-level control")
    print("=" * 80)
    print()

    # Save configuration
    config_file = log_dir / "config.txt"
    with open(config_file, "w") as f:
        f.write("Adaptive Gait Training Configuration\n")
        f.write("=" * 40 + "\n")
        for key, value in asdict(cfg).items():
            f.write(f"{key}: {value}\n")

    # Create vectorized environment
    print("\nCreating environments...")
    env_fns = [make_env(log_dir, i, cfg) for i in range(cfg.n_envs)]
    if cfg.vec_env_backend == "subproc" and cfg.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    # Wrap with VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Get network architecture
    policy_kwargs = get_network_architecture(cfg.network_size)

    print(f"\nNetwork architecture ({cfg.network_size}):")
    print(f"  Actor:  {policy_kwargs['net_arch']['pi']}")
    print(f"  Critic: {policy_kwargs['net_arch']['vf']}")
    print(f"  Activation: {policy_kwargs['activation_fn'].__name__}")

    # PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        n_epochs=cfg.n_epochs,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=str(log_dir),
        verbose=1,
        device=cfg.device,
        seed=cfg.seed,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq // cfg.n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="rl_model",
        save_vecnormalize=True,
    )

    # Print training info
    print("\nTraining details:")
    print(f"  Steps per iteration:   {cfg.n_steps * cfg.n_envs:,}")
    print(f"  Updates per iteration: {cfg.n_epochs * (cfg.n_steps * cfg.n_envs // cfg.batch_size):,}")
    print(f"  Total iterations:      {cfg.total_timesteps // (cfg.n_steps * cfg.n_envs):,}")
    print(f"  Checkpoint frequency:  Every {cfg.checkpoint_freq:,} steps")
    print()

    # Train
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    print()
    print("Monitor training progress:")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print(f"  Checkpoints: {log_dir}/checkpoints/")
    print()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final artifacts
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    model.save(str(log_dir / "final_model"))
    vec_env.save(str(log_dir / "vec_normalize.pkl"))

    print(f"\n{'=' * 80}")
    print("Training complete!")
    print(f"{'=' * 80}")
    print(f"\nArtifacts saved to: {log_dir}")
    print("  ✓ final_model.zip")
    print("  ✓ vec_normalize.pkl")
    print("  ✓ checkpoints/")
    print("  ✓ monitor_*.csv")
    print("  ✓ config.txt")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("=" * 80)
    print("\n1. Test the trained policy:")
    print(f"   python3 play_adaptive_policy.py \\")
    print(f"       --model {log_dir}/final_model.zip \\")
    print(f"       --normalize {log_dir}/vec_normalize.pkl \\")
    print(f"       --seconds 30 \\")
    print(f"       --deterministic")
    print()
    print("   Optional flags:")
    print("     --flat          Use flat terrain instead of rough")
    print("     --no-reset      Disable auto-reset on termination")
    print()
    print("\n2. Monitor training metrics:")
    print(f"   tensorboard --logdir {log_dir}")
    print()
    print("\n3. Analyze training results:")
    print(f"   python3 analyze_training_results.py {log_dir}")
    print(f"   python3 plot_learning_curve.py {log_dir}")
    print()
    print("Expected behavior:")
    print("  - Policy should increase step_height on rough terrain")
    print("  - Policy should adjust step_length for speed/stability trade-off")
    print("  - Policy should modulate cycle_time based on terrain difficulty")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
