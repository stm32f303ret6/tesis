#!/usr/bin/env python3
"""Compare residual policy vs baseline (zero residuals) on short rollouts.

Generates phases_test/residual_vs_baseline.png with two panels:
- Mean distance traveled
- Fall rate (fraction of episodes terminated early)

Usage
  python3 tests/compare_residual_vs_baseline.py \
    --model runs/smoke4_XXXX/final_model.zip \
    --normalize runs/smoke4_XXXX/vec_normalize.pkl \
    --episodes 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym  # noqa: F401
except Exception:
    import gym  # type: ignore  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.residual_walk_env import ResidualWalkEnv
from gait_controller import GaitParameters


def make_env() -> ResidualWalkEnv:
    # Use same gait parameters as training
    gait = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )
    return ResidualWalkEnv(model_path="model/world_train.xml", gait_params=gait)


def load_vecnormalize(path: Optional[str | Path]) -> Optional[VecNormalize]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {p}")
    dummy = DummyVecEnv([make_env])
    vn = VecNormalize.load(str(p), dummy)
    vn.training = False
    vn.norm_reward = False
    return vn


def run_mode(episodes: int, model: Optional[PPO], vn: Optional[VecNormalize]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run episodes and return arrays: distances, heights_mean, falls(0/1)."""
    def _env_fn():
        return make_env()

    vec = DummyVecEnv([_env_fn])
    if vn is not None:
        vn.venv = vec
        vec = vn
        vec.training = False
        vec.norm_reward = False

    distances = []
    heights = []
    falls = []

    for _ in range(episodes):
        reset_out = vec.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out
        done = False
        truncated = False
        distance = 0.0
        step_heights = []

        # Access base env for sensors and dt
        base_env = None
        try:
            if isinstance(vec, VecNormalize):  # type: ignore[arg-type]
                base_env = vec.venv.envs[0]  # type: ignore[attr-defined]
            else:
                base_env = vec.envs[0]  # type: ignore[attr-defined]
        except Exception:
            base_env = None
        dt = float(getattr(getattr(base_env, "model", object()), "opt", object()).timestep) if base_env is not None else 0.0

        while not (done or truncated):
            if model is None:
                action = np.zeros((1, vec.action_space.shape[0]), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray) and action.ndim == 1:
                    action = action[None, ...]
            step_out = vec.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 4:
                obs, reward, done_flag, info = step_out
                info0 = info[0] if isinstance(info, (list, tuple)) and info else info
                truncated = bool(getattr(info0, "get", lambda *_: False)("TimeLimit.truncated", False)) if isinstance(info0, dict) else False
                done = bool(done_flag)
            else:
                obs, reward, done, truncated, info = step_out  # type: ignore[misc]
            # Height and distance
            info0 = info[0] if isinstance(info, (list, tuple)) and info else info
            if isinstance(info0, dict):
                step_heights.append(float(info0.get("body_height", 0.0)))
            if base_env is not None:
                try:
                    fv = float(base_env.sensor_reader.read_sensor("body_linvel")[0])
                    if dt > 0.0:
                        distance += fv * dt
                except Exception:
                    pass

        distances.append(distance)
        heights.append(np.mean(step_heights) if step_heights else 0.0)
        falls.append(1.0 if bool(done) else 0.0)

    return np.array(distances), np.array(heights), np.array(falls)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare residual policy vs baseline")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--normalize", type=str, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("phases_test") / "residual_vs_baseline.png"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path("phases_test").mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    vn = load_vecnormalize(args.normalize)

    dist_b, h_b, fall_b = run_mode(args.episodes, model=None, vn=vn)
    dist_r, h_r, fall_r = run_mode(args.episodes, model=model, vn=vn)

    # Aggregates
    mean_dist = [float(np.mean(dist_b)), float(np.mean(dist_r))]
    fall_rate = [float(np.mean(fall_b)), float(np.mean(fall_r))]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    labels = ["baseline", "residual"]
    ax1.bar(labels, mean_dist, color=["#888", "#1f77b4"])  # type: ignore[arg-type]
    ax1.set_title("Mean distance (m)")
    ax1.set_ylabel("meters")

    ax2.bar(labels, fall_rate, color=["#888", "#1f77b4"])  # type: ignore[arg-type]
    ax2.set_title("Fall rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("fraction of episodes")

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
