#!/usr/bin/env python3
"""Visualize and evaluate trained adaptive gait policy.

This script loads a trained adaptive gait policy and runs it in the MuJoCo
viewer, displaying real-time gait parameter adaptations.

Usage:
    python3 play_adaptive_policy.py \\
        --model runs/adaptive_gait_XXX/final_model.zip \\
        --normalize runs/adaptive_gait_XXX/vec_normalize.pkl \\
        --seconds 30 \\
        --deterministic
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.adaptive_gait_env import AdaptiveGaitEnv
from gait_controller import GaitParameters


def make_env():
    """Create evaluation environment."""
    gait = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )
    return AdaptiveGaitEnv(
        model_path="model/world_train.xml",
        gait_params=gait,
        residual_scale=0.01,
        max_episode_steps=10000,
        settle_steps=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Play trained adaptive gait policy")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--normalize", type=str, default=None, help="Path to VecNormalize stats (.pkl)")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duration to run [seconds]")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--flat", action="store_true", help="Use flat terrain instead of rough")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    # Create environment
    print("Creating environment...")
    env = make_env()

    # Override with flat terrain if requested
    if args.flat:
        print("Using flat terrain (world.xml)")
        env = AdaptiveGaitEnv(
            model_path="model/world.xml",
            gait_params=GaitParameters(
                body_height=0.05,
                step_length=0.06,
                step_height=0.04,
                cycle_time=0.8
            ),
            residual_scale=0.01,
            max_episode_steps=10000,
            settle_steps=0,
        )

    # Wrap for normalization if stats provided
    if args.normalize:
        print(f"Loading normalization stats from {args.normalize}...")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False  # Disable running mean updates
        vec_env.norm_reward = False  # Don't normalize rewards during eval
        use_vec_env = True
    else:
        print("No normalization stats provided, using raw observations")
        use_vec_env = False

    print("\n" + "=" * 80)
    print("Starting playback...")
    print("=" * 80)
    print(f"Duration:        {args.seconds:.1f}s")
    print(f"Deterministic:   {args.deterministic}")
    print(f"Terrain:         {'flat (world.xml)' if args.flat else 'rough (world_train.xml)'}")
    print("\nWatch the console for real-time gait parameter updates!")
    print("=" * 80)
    print()

    # Open MuJoCo viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Reset environment
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        start_time = time.time()
        step_count = 0
        last_print_time = start_time
        print_interval = 1.0  # Print gait params every 1 second

        # Tracking for statistics
        gait_param_history = {
            "step_height": [],
            "step_length": [],
            "cycle_time": [],
            "body_height": [],
        }

        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.seconds:
                break

            # Get action from policy
            if use_vec_env:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            # Step environment
            if use_vec_env:
                obs, reward, done, info = vec_env.step(action)
                # Extract info from vec env
                if done[0]:
                    obs = vec_env.reset()
                current_info = vec_env.env_method("step", action[0])[0][-1]  # Get info
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    obs, _ = env.reset()
                current_info = info

            # Update viewer
            viewer.sync()

            # Print gait parameters periodically
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                if "gait_params" in current_info:
                    params = current_info["gait_params"]
                    print(f"[t={elapsed:.1f}s] Gait params: "
                          f"step_h={params['step_height']:.4f}m, "
                          f"step_l={params['step_length']:.4f}m, "
                          f"cycle_t={params['cycle_time']:.3f}s, "
                          f"body_h={params['body_height']:.4f}m")

                    # Track history
                    for key in gait_param_history:
                        gait_param_history[key].append(params[key])

                last_print_time = current_time

            step_count += 1

            # Control playback speed
            time.sleep(env.model.opt.timestep)

    # Print statistics
    print("\n" + "=" * 80)
    print("Playback complete!")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Duration:    {elapsed:.1f}s")
    print()

    if gait_param_history["step_height"]:
        print("Gait Parameter Statistics:")
        print("-" * 40)
        for param_name, values in gait_param_history.items():
            arr = np.array(values)
            print(f"{param_name:15s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}")
        print()
        print("Interpretation:")
        print("  - Large std indicates adaptive behavior (good for rough terrain)")
        print("  - Small std indicates policy relies mostly on base parameters")
        print("  - Compare with base gait params to see adaptation magnitude")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
