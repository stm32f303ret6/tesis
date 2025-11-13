#!/usr/bin/env python3
"""Play a trained residual PPO policy in the MuJoCo viewer for N seconds.

Example
  python3 phases_test/phase4_viewer_play_policy.py \
    --model runs/smoke4_20251111_154549/final_model.zip \
    --normalize runs/smoke4_20251111_154549/vec_normalize.pkl \
    --seconds 20 --deterministic

Baseline (zero residuals)
  python3 phases_test/phase4_viewer_play_policy.py --baseline --seconds 20

Notes
- Forces MUJOCO_GL=glfw to open a window (override with --gl-backend).
- Close the viewer window (ESC) to stop early.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional
import time

import numpy as np
try:
    import mujoco  # type: ignore
except Exception:  # pragma: no cover - optional dependency for ID lookup
    mujoco = None  # type: ignore

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


def make_env(residual_scale: float) -> ResidualWalkEnv:
    """Create the playback environment.

    Matches Phase 2 gait parameters and v3 training world. Allows overriding
    residual scale to match training runs (default 0.10 in v3 training).
    """
    gait = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
    return ResidualWalkEnv(
        model_path="model/world_train.xml",
        gait_params=gait,
        residual_scale=float(residual_scale),
    )


def load_vecnormalize(path: Optional[str | Path], base_vec) -> Optional[VecNormalize]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {p}")
    vn = VecNormalize.load(str(p), base_vec)
    vn.training = False
    vn.norm_reward = False
    return vn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Viewer playback of residual PPO policy")
    p.add_argument("--model", type=str, default=None, help="Path to SB3 .zip model")
    p.add_argument("--normalize", type=str, default=None, help="Path to VecNormalize .pkl")
    p.add_argument("--seconds", type=float, default=20.0, help="Simulation seconds to run")
    p.add_argument("--deterministic", action="store_true", help="Deterministic policy actions")
    p.add_argument("--baseline", action="store_true", help="Use zero actions (no residuals)")
    p.add_argument("--gl-backend", type=str, default="glfw", help="MUJOCO_GL backend (glfw/egl/osmesa)")
    p.add_argument("--residual-scale", type=float, default=0.01, help="Residual action scale used during training (e.g., 0.10)")
    p.add_argument("--settle-steps", type=int, default=0, help="Override env settle steps during reset")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure windowed GL unless user overrides
    os.environ["MUJOCO_GL"] = args.gl_backend

    # Build single-env VecEnv so we can reuse normalization and SB3 model
    vec = DummyVecEnv([lambda: make_env(args.residual_scale)])

    # Log key playback parameters for sanity
    print(f"[viewer] MUJOCO_GL={args.gl_backend}  residual_scale={args.residual_scale}")

    vn = load_vecnormalize(args.normalize, vec)
    if vn is not None:
        vec = vn

    model: Optional[PPO] = None
    if not args.baseline:
        if args.model is None:
            raise SystemExit("--model is required unless --baseline is set")
        model = PPO.load(args.model, device="cpu")

    # Access the underlying base environment for viewer/model/data
    base_env = None
    try:
        if isinstance(vec, VecNormalize):  # type: ignore[arg-type]
            base_env = vec.venv.envs[0]  # type: ignore[attr-defined]
        else:
            base_env = vec.envs[0]  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError("Unable to access base environment from VecEnv") from exc

    # Launch viewer
    viewer = None
    headless = (args.gl_backend.lower() != "glfw")
    if not headless:
        try:
            import mujoco.viewer as mj_viewer  # type: ignore[attr-defined]
            viewer = mj_viewer.launch_passive(base_env.model, base_env.data)
            # Configure camera: close follow view
            viewer.cam.distance = 0.5
            viewer.cam.azimuth = 100
            viewer.cam.elevation = -10
        except Exception:
            print("[WARN] MuJoCo viewer unavailable or failed to launch; running headless.")
            viewer = None

    # Reset env (through vec wrapper to keep normalization consistent)
    reset_out = vec.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out

    # Determine step count from simulation dt
    dt = float(base_env.model.opt.timestep)
    steps = max(1, int(args.seconds / max(1e-9, dt)))
    # Prevent premature time-limit truncation by extending max_episode_steps beyond planned steps
    try:
        base_env.max_episode_steps = max(int(getattr(base_env, "max_episode_steps", 1000)), steps + 5)
        base_env.settle_steps = int(args.settle_steps)
    except Exception:
        pass

    # Prepare camera follow and body id lookup (works headless too)
    robot_body_id = None
    if mujoco is not None:
        try:
            robot_body_id = mujoco.mj_name2id(
                base_env.model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
        except Exception:
            robot_body_id = None

    # Rollout loop — sync viewer every step at real-time rate
    for _ in range(steps):
        step_start = time.time()
        if viewer is not None and getattr(viewer, "is_running", lambda: True)() is False:
            break
        if model is None:
            action = np.zeros((1, base_env.action_space.shape[0]), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            if isinstance(action, np.ndarray) and action.ndim == 1:
                action = action[None, ...]
        step_out = vec.step(action)
        # Support both SB3 VecEnv 4-tuple and Gymnasium 5-tuple
        if isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done_flag, info = step_out
            # If episode ended, auto-reset via vec
            if bool(done_flag):
                reset_out = vec.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out
        else:
            obs, reward, done, truncated, info = step_out  # type: ignore[misc]
            if bool(done) or bool(truncated):
                reset_out = vec.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out
        if viewer is not None:
            try:
                if robot_body_id is not None:
                    viewer.cam.lookat[:] = base_env.data.xpos[robot_body_id]
            except Exception:
                pass
            viewer.sync()
        # Maintain real-time pacing
        remaining = dt - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    # Print final body/world position before quitting
    try:
        if robot_body_id is not None:
            final_pos = np.array(base_env.data.xpos[robot_body_id]).copy()
            # World frame position in meters
            print(f"[viewer] final 'robot' body world position: {final_pos}")
        else:
            # Fallback: root translation from qpos if available
            if getattr(base_env.model, "nq", 0) >= 3:
                root_pos = np.array(base_env.data.qpos[:3]).copy()
                print(f"[viewer] final root qpos translation: {root_pos}")
    except Exception:
        pass

    if viewer is not None:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
