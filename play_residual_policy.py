#!/usr/bin/env python3
"""Play back a trained residual PPO policy or run a baseline with zero actions.

Usage examples
- Residual policy (with normalization):
    python3 play_residual_policy.py \
        --model runs/smoke4_XXXX/final_model.zip \
        --normalize runs/smoke4_XXXX/vec_normalize.pkl \
        --episodes 3

- Baseline gait only (zero residuals):
    python3 play_residual_policy.py --episodes 3 --baseline

Outputs
- Prints per-episode stats and writes an optional summary file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym  # noqa: F401
except Exception:  # pragma: no cover - fallback if gymnasium is absent
    import gym  # type: ignore  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_walk_env import ResidualWalkEnv
from gait_controller import GaitParameters


@dataclass
class EpisodeStats:
    reward: float
    steps: int
    distance: float
    avg_height: float
    avg_forward_vel: float
    terminated: bool
    truncated: bool


def make_env() -> ResidualWalkEnv:
    # Match Phase 2/height_control gait parameters for visual parity
    gait = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
    return ResidualWalkEnv(model_path="model/world_train.xml", gait_params=gait)


def run_episodes(
    episodes: int,
    model: Optional[PPO] = None,
    vec_norm: Optional[VecNormalize] = None,
    deterministic: bool = True,
) -> Tuple[EpisodeStats, ...]:
    # Build VecEnv for optional VecNormalize compatibility
    def _env_fn():
        return make_env()

    vec = DummyVecEnv([_env_fn])
    if vec_norm is not None:
        # Use provided normalizer with frozen stats for evaluation
        vec_norm.venv = vec  # ensure underlying env
        vec = vec_norm
        vec.training = False
        vec.norm_reward = False

    stats: list[EpisodeStats] = []
    for _ in range(episodes):
        reset_out = vec.reset()
        # SB3 VecEnv.reset() returns obs only; handle both protocols
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs = reset_out[0]
        else:
            obs = reset_out
        done = False
        truncated = False
        ep_rew = 0.0
        ep_steps = 0
        heights = []
        fwd_vels = []
        distance = 0.0

        # Access underlying single environment for sensors and dt
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
                action, _ = model.predict(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim == 1:
                    action = action[None, ...]
            step_out = vec.step(action)
            # SB3 VecEnv.step() returns (obs, rewards, dones, infos)
            if isinstance(step_out, tuple) and len(step_out) == 4:
                obs, reward, done_flag, info = step_out
                # Try to recover truncated flag from info if present
                info0 = info[0] if isinstance(info, (list, tuple)) and info else info
                truncated = bool(getattr(info0, "get", lambda *_: False)("TimeLimit.truncated", False)) if isinstance(info0, dict) else False
                done = bool(done_flag)
            else:
                # Fallback to Gymnasium-style 5-tuple
                obs, reward, done, truncated, info = step_out  # type: ignore[misc]
            ep_rew += float(reward)
            ep_steps += 1
            # Unwrap info: VecEnv returns a list
            info0 = info[0] if isinstance(info, (list, tuple)) and info else info
            if isinstance(info0, dict):
                heights.append(float(info0.get("body_height", 0.0)))
            # Prefer sensor-derived forward velocity from base env
            if base_env is not None:
                try:
                    fv = float(base_env.sensor_reader.read_sensor("body_linvel")[0])
                    fwd_vels.append(fv)
                    if dt > 0.0:
                        distance += fv * dt
                except Exception:
                    pass
        avg_h = float(np.mean(heights)) if heights else 0.0
        avg_v = float(np.mean(fwd_vels)) if fwd_vels else 0.0
        stats.append(
            EpisodeStats(
                reward=float(ep_rew),
                steps=int(ep_steps),
                distance=float(distance),
                avg_height=avg_h,
                avg_forward_vel=avg_v,
                terminated=bool(done),
                truncated=bool(truncated),
            )
        )
    return tuple(stats)


def load_vecnormalize(path: Optional[str | Path]) -> Optional[VecNormalize]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {p}")
    # Dummy env required by load; replaced later
    dummy = DummyVecEnv([make_env])
    vn = VecNormalize.load(str(p), dummy)
    vn.training = False
    vn.norm_reward = False
    return vn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual PPO policy playback")
    p.add_argument("--model", type=str, default=None, help="Path to SB3 .zip model")
    p.add_argument(
        "--normalize", type=str, default=None, help="Path to VecNormalize .pkl"
    )
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--baseline", action="store_true", help="Run zero-action baseline")
    p.add_argument(
        "--summary-out",
        type=str,
        default=str(Path("phases_test") / "phase4_playback.txt"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    Path("phases_test").mkdir(parents=True, exist_ok=True)

    model: Optional[PPO] = None
    vn: Optional[VecNormalize] = None

    if not args.baseline:
        if args.model is None:
            raise SystemExit("--model is required unless --baseline is set")
        vn = load_vecnormalize(args.normalize)
        model = PPO.load(args.model, device="cpu")
    else:
        vn = load_vecnormalize(args.normalize)

    stats = run_episodes(args.episodes, model=model, vec_norm=vn, deterministic=args.deterministic)

    # Print and write summary
    lines = []
    mode = "baseline_zero" if args.baseline else "residual_policy"
    for i, s in enumerate(stats):
        line = (
            f"episode={i} mode={mode} reward={s.reward:.3f} steps={s.steps} "
            f"distance={s.distance:.3f} avg_height={s.avg_height:.4f} avg_fwd_vel={s.avg_forward_vel:.4f} "
            f"terminated={s.terminated} truncated={s.truncated}"
        )
        print(line)
        lines.append(line)
    out_path = Path(args.summary_out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Summary saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
