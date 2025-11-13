#!/usr/bin/env python3
"""Plot gait parameter adaptation over time.

Usage:
    python3 plot_gait_adaptation.py \\
        --model runs/adaptive_gait_XXX/final_model.zip \\
        --normalize runs/adaptive_gait_XXX/vec_normalize.pkl \\
        --seconds 60
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.adaptive_gait_env import AdaptiveGaitEnv
from gait_controller import GaitParameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--normalize", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    # Create environment
    print("Creating environment...")
    gait = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )
    env = AdaptiveGaitEnv(
        model_path="model/world_train.xml",
        gait_params=gait,
        residual_scale=0.01,
        max_episode_steps=10000,
        settle_steps=0,
    )

    # Wrap for normalization
    if args.normalize:
        print(f"Loading normalization stats...")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec_env = True
    else:
        use_vec_env = False

    # Run policy and collect data
    print(f"Running policy for {args.seconds}s...")
    if use_vec_env:
        obs = vec_env.reset()
    else:
        obs, _ = env.reset()

    # Data collection
    timestamps = []
    gait_params = {
        "step_height": [],
        "step_length": [],
        "cycle_time": [],
        "body_height": [],
    }
    body_positions = []
    rewards = []

    start_time = time.time()
    step_count = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= args.seconds:
            break

        # Get action
        if use_vec_env:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _ = vec_env.step(action)
            if done[0]:
                obs = vec_env.reset()
            # Get info from underlying env
            info = env.controller.get_current_parameters()
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info_dict = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            info = info_dict.get("gait_params", env.controller.get_current_parameters())
            reward = float(reward)

        # Record data
        timestamps.append(elapsed)
        for key in gait_params:
            gait_params[key].append(info[key])

        body_pos = env.sensor_reader.read_sensor("body_pos")
        body_positions.append(body_pos[0])  # X position
        rewards.append(reward)

        step_count += 1

        # Small delay for real-time visualization
        if step_count % 10 == 0:
            print(f"\r[t={elapsed:.1f}s] Steps: {step_count}", end="", flush=True)

    print(f"\n\nCollected {step_count} steps over {elapsed:.1f}s")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Gait Parameter Evolution", fontsize=16, fontweight="bold")

    # Plot gait parameters
    param_axes = axes[:2].flatten()
    for ax, (name, values) in zip(param_axes, gait_params.items()):
        ax.plot(timestamps, values, linewidth=1.5, alpha=0.8)
        ax.set_ylabel(name.replace("_", " ").title() + " [m/s]", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]", fontsize=10)

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color="red", linestyle="--", alpha=0.5, label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=9)

    # Plot forward distance
    axes[2, 0].plot(timestamps, body_positions, linewidth=1.5, color="green")
    axes[2, 0].set_ylabel("Forward Distance [m]", fontsize=11)
    axes[2, 0].set_xlabel("Time [s]", fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    avg_velocity = body_positions[-1] / timestamps[-1]
    axes[2, 0].set_title(f"Avg Velocity: {avg_velocity:.3f} m/s", fontsize=10)

    # Plot reward
    axes[2, 1].plot(timestamps, rewards, linewidth=1, alpha=0.7, color="purple")
    axes[2, 1].set_ylabel("Reward", fontsize=11)
    axes[2, 1].set_xlabel("Time [s]", fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title(f"Mean Reward: {np.mean(rewards):.1f}", fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = "gait_adaptation_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_file}")

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
