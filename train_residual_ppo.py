#!/usr/bin/env python3
"""PPO training script for residual-learning locomotion.

Uses Gymnasium + Stable-Baselines3 with VecNormalize and TensorBoard logging.
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

from envs.residual_walk_env import ResidualWalkEnv


def make_env(log_dir: Path, rank: int):
    def _init():
        env = ResidualWalkEnv()
        # Monitor writes per-episode stats; helps SB3 compute ep_rew_mean
        monitor_file = log_dir / f"monitor_{rank}.csv"
        env = Monitor(env, filename=str(monitor_file))
        return env

    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=10000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--run-name", type=str, default="smoke")
    p.add_argument("--log-root", type=str, default="runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_root) / f"{args.run_name}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # VecEnv creation
    env_fns = [make_env(log_dir, i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO configuration
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    # Train
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    # Save artifacts
    model.save(str(log_dir / "final_model"))
    vec_env.save(str(log_dir / "vec_normalize.pkl"))

    print(f"Training complete. Artifacts saved under {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

