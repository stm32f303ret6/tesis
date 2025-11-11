#!/usr/bin/env python3
"""Phase 2 smoke test for ResidualWalkEnv.

Runs 100 steps with zero actions, then 100 with random actions. Logs:
- phases_test/phase2_env_spec.txt
- phases_test/phase2_step_trace.csv
- phases_test/phase2_reward_components.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym  # noqa: F401
except Exception:
    try:
        import gym  # noqa: F401
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.residual_walk_env import ResidualWalkEnv


OUT_DIR = Path(__file__).parent


def main() -> int:
    # Ensure headless GL for MuJoCo if unset
    os.environ.setdefault("MUJOCO_GL", "egl")

    env = ResidualWalkEnv()

    # Spec output
    obs, _ = env.reset()
    obs_shape = tuple(obs.shape)
    act_shape = tuple(env.action_space.shape)
    (OUT_DIR / "phase2_env_spec.txt").write_text(
        f"observation_shape: {obs_shape}\naction_shape: {act_shape}\n",
        encoding="utf-8",
    )

    # Step trace and reward components
    trace_path = OUT_DIR / "phase2_step_trace.csv"
    comp_path = OUT_DIR / "phase2_reward_components.json"
    with trace_path.open("w", encoding="utf-8") as f:
        f.write("step,total_reward,forward_velocity,body_height,terminated,truncated\n")

        # Accumulate reward components
        comp_sums = {}
        comp_count = 0

        # 100 zero-action steps
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for i in range(100):
            obs, rew, term, trunc, info = env.step(zero_action)
            assert np.all(np.isfinite(obs)), "Non-finite values in observation"
            assert np.isfinite(rew), "Non-finite reward"
            fv = float(env.sensor_reader.read_sensor("body_linvel")[0])
            bh = float(env.sensor_reader.read_sensor("body_pos")[2])
            f.write(f"{i},{rew:.6f},{fv:.6f},{bh:.6f},{int(term)},{int(trunc)}\n")

            comps = info.get("reward_components", {})
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
            comp_count += 1

        # 100 random-action steps
        for i in range(100, 200):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)), "Non-finite values in observation"
            assert np.isfinite(rew), "Non-finite reward"
            fv = float(env.sensor_reader.read_sensor("body_linvel")[0])
            bh = float(env.sensor_reader.read_sensor("body_pos")[2])
            f.write(f"{i},{rew:.6f},{fv:.6f},{bh:.6f},{int(term)},{int(trunc)}\n")

            comps = info.get("reward_components", {})
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
            comp_count += 1

    # Mean reward components
    comp_means = {k: (v / max(1, comp_count)) for k, v in comp_sums.items()}
    comp_path.write_text(json.dumps(comp_means, indent=2), encoding="utf-8")

    print("Phase 2 smoke complete. Artifacts written:")
    print(f"- {trace_path}")
    print(f"- {comp_path}")
    print(f"- {OUT_DIR / 'phase2_env_spec.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
