#!/usr/bin/env python3
"""Compare residual-only vs adaptive gait approaches.

This script runs both approaches on the same terrain and compares:
1. Episode length (how long before falling)
2. Forward distance traveled
3. Velocity tracking error
4. Stability (roll/pitch variance)

Usage:
    python3 compare_approaches.py \\
        --residual-model runs/prod_v3_XXX/final_model.zip \\
        --residual-normalize runs/prod_v3_XXX/vec_normalize.pkl \\
        --adaptive-model runs/adaptive_gait_XXX/final_model.zip \\
        --adaptive-normalize runs/adaptive_gait_XXX/vec_normalize.pkl \\
        --episodes 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_walk_env import ResidualWalkEnv
from envs.adaptive_gait_env import AdaptiveGaitEnv
from gait_controller import GaitParameters


def evaluate_policy(
    model: PPO,
    env,
    vec_env,
    n_episodes: int,
    deterministic: bool = True,
) -> Dict[str, List[float]]:
    """Evaluate policy and collect statistics."""
    results = {
        "episode_lengths": [],
        "distances": [],
        "avg_velocities": [],
        "roll_stds": [],
        "pitch_stds": [],
        "rewards": [],
    }

    for ep in range(n_episodes):
        obs = vec_env.reset() if vec_env else env.reset()[0]
        done = False
        step_count = 0
        total_reward = 0.0

        # Tracking
        initial_pos = env.sensor_reader.read_sensor("body_pos")[0]
        rolls = []
        pitches = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            if vec_env:
                obs, reward, done_arr, _ = vec_env.step(action)
                done = done_arr[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            step_count += 1
            total_reward += reward

            # Track stability
            from envs.residual_walk_env import quat_to_euler
            quat = env.sensor_reader.get_body_quaternion()
            roll, pitch, _ = quat_to_euler(quat, False)
            rolls.append(roll)
            pitches.append(pitch)

            # Limit episode length for comparison
            if step_count >= 5000:
                break

        # Compute metrics
        final_pos = env.sensor_reader.read_sensor("body_pos")[0]
        distance = final_pos - initial_pos
        avg_velocity = distance / (step_count * env.model.opt.timestep)

        results["episode_lengths"].append(step_count)
        results["distances"].append(distance)
        results["avg_velocities"].append(avg_velocity)
        results["roll_stds"].append(np.std(rolls))
        results["pitch_stds"].append(np.std(pitches))
        results["rewards"].append(total_reward)

        print(f"  Episode {ep+1}/{n_episodes}: "
              f"steps={step_count}, dist={distance:.2f}m, "
              f"vel={avg_velocity:.3f}m/s, reward={total_reward:.1f}")

    return results


def print_statistics(name: str, results: Dict[str, List[float]]) -> None:
    """Print summary statistics."""
    print(f"\n{name} Results:")
    print("=" * 60)
    for metric, values in results.items():
        arr = np.array(values)
        print(f"  {metric:20s}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare residual-only vs adaptive gait")
    parser.add_argument("--residual-model", type=str, required=True)
    parser.add_argument("--residual-normalize", type=str, default=None)
    parser.add_argument("--adaptive-model", type=str, required=True)
    parser.add_argument("--adaptive-normalize", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--terrain", type=str, default="rough", choices=["rough", "flat"])
    args = parser.parse_args()

    terrain_file = "model/world_train.xml" if args.terrain == "rough" else "model/world.xml"

    print("=" * 80)
    print("Comparing Residual-Only vs Adaptive Gait Approaches")
    print("=" * 80)
    print(f"Terrain:       {args.terrain} ({terrain_file})")
    print(f"Episodes:      {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 80)

    # Setup base gait parameters
    gait = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )

    # ===================== Evaluate Residual-Only =====================
    print("\n[1/2] Evaluating Residual-Only Policy...")
    print("-" * 60)

    residual_model = PPO.load(args.residual_model)
    residual_env = ResidualWalkEnv(
        model_path=terrain_file,
        gait_params=gait,
        residual_scale=0.01,
        max_episode_steps=5000,
        settle_steps=0,
    )

    if args.residual_normalize:
        residual_vec_env = DummyVecEnv([lambda: residual_env])
        residual_vec_env = VecNormalize.load(args.residual_normalize, residual_vec_env)
        residual_vec_env.training = False
        residual_vec_env.norm_reward = False
    else:
        residual_vec_env = None

    residual_results = evaluate_policy(
        residual_model,
        residual_env,
        residual_vec_env,
        args.episodes,
        args.deterministic,
    )

    # ===================== Evaluate Adaptive Gait =====================
    print("\n[2/2] Evaluating Adaptive Gait Policy...")
    print("-" * 60)

    adaptive_model = PPO.load(args.adaptive_model)
    adaptive_env = AdaptiveGaitEnv(
        model_path=terrain_file,
        gait_params=gait,
        residual_scale=0.01,
        max_episode_steps=5000,
        settle_steps=0,
    )

    if args.adaptive_normalize:
        adaptive_vec_env = DummyVecEnv([lambda: adaptive_env])
        adaptive_vec_env = VecNormalize.load(args.adaptive_normalize, adaptive_vec_env)
        adaptive_vec_env.training = False
        adaptive_vec_env.norm_reward = False
    else:
        adaptive_vec_env = None

    adaptive_results = evaluate_policy(
        adaptive_model,
        adaptive_env,
        adaptive_vec_env,
        args.episodes,
        args.deterministic,
    )

    # ===================== Print Comparison =====================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print_statistics("Residual-Only", residual_results)
    print_statistics("Adaptive Gait", adaptive_results)

    # Compute improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENTS (Adaptive vs Residual-Only)")
    print("=" * 80)

    improvements = {}
    for metric in residual_results.keys():
        residual_mean = np.mean(residual_results[metric])
        adaptive_mean = np.mean(adaptive_results[metric])

        if "std" in metric:
            # Lower is better for stability metrics
            improvement = (residual_mean - adaptive_mean) / residual_mean * 100
            improvements[metric] = improvement
            print(f"  {metric:20s}: {improvement:+.1f}% "
                  f"({'better' if improvement > 0 else 'worse'})")
        else:
            # Higher is better for performance metrics
            improvement = (adaptive_mean - residual_mean) / abs(residual_mean) * 100
            improvements[metric] = improvement
            print(f"  {metric:20s}: {improvement:+.1f}% "
                  f"({'better' if improvement > 0 else 'worse'})")

    # Overall verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Count number of improved metrics
    improved_count = sum(1 for imp in improvements.values() if imp > 1.0)
    total_metrics = len(improvements)

    if improved_count >= total_metrics * 0.6:
        print("✓ Adaptive Gait shows significant improvement over Residual-Only")
        print(f"  ({improved_count}/{total_metrics} metrics improved by >1%)")
    elif improved_count >= total_metrics * 0.4:
        print("≈ Adaptive Gait shows modest improvement over Residual-Only")
        print(f"  ({improved_count}/{total_metrics} metrics improved by >1%)")
    else:
        print("✗ Residual-Only performs comparably or better than Adaptive Gait")
        print(f"  ({improved_count}/{total_metrics} metrics improved by >1%)")
        print("\nPossible reasons:")
        print("  - Adaptive policy undertrained (needs more timesteps)")
        print("  - Terrain not rough enough to benefit from adaptation")
        print("  - Parameter delta scales too small (policy can't adapt enough)")
        print("  - Reward doesn't encourage parameter adaptation")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
