#!/usr/bin/env python3
"""Debug script to compare baseline vs trained model performance.

Usage:
    python3 debug_model.py runs/prod_5m_20251111_173611
"""

import sys
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_walk_env import ResidualWalkEnv
from gait_controller import GaitParameters


def test_policy(model, vn, steps=500, label="Policy"):
    """Run policy for N steps and return stats."""
    obs = vn.reset()
    base_env = vn.venv.envs[0]

    rewards = []
    positions = []
    heights = []
    fwd_vels = []
    actions_mag = []

    for i in range(steps):
        if model is None:
            action = np.zeros((1, base_env.action_space.shape[0]), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
            if action.ndim == 1:
                action = action[None, ...]

        actions_mag.append(float(np.linalg.norm(action)))
        obs, reward, done, info = vn.step(action)

        rewards.append(float(reward[0]))
        body_pos = base_env.sensor_reader.read_sensor("body_pos")
        linvel = base_env.sensor_reader.read_sensor("body_linvel")

        positions.append(float(body_pos[0]))
        heights.append(float(body_pos[2]))
        fwd_vels.append(float(linvel[0]))

        if done[0]:
            print(f"  {label} terminated at step {i}")
            break

    return {
        "label": label,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "final_position": positions[-1],
        "mean_height": float(np.mean(heights)),
        "mean_fwd_vel": float(np.mean(fwd_vels)),
        "mean_action_mag": float(np.mean(actions_mag)),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 debug_model.py <run_dir>")
        print("Example: python3 debug_model.py runs/prod_5m_20251111_173611")
        return 1

    run_dir = Path(sys.argv[1])
    model_path = run_dir / "final_model.zip"
    norm_path = run_dir / "vec_normalize.pkl"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    print("="*70)
    print(f"Debugging: {run_dir.name}")
    print("="*70)

    # Load model
    model = PPO.load(str(model_path), device="cpu")

    # Test 1: Training gait (default)
    print("\n[Test 1] Training Gait (default parameters)")
    print("-"*70)

    def make_env_train():
        return ResidualWalkEnv(model_path="model/world_train.xml")

    vec1 = DummyVecEnv([make_env_train])
    vn1 = VecNormalize.load(str(norm_path), vec1)
    vn1.training = False
    vn1.norm_reward = False

    print("Baseline (zero residuals):")
    baseline_train = test_policy(None, vn1, steps=500, label="Baseline")
    print(f"  Distance: {baseline_train['final_position']:.4f}m")
    print(f"  Mean velocity: {baseline_train['mean_fwd_vel']:.4f} m/s")
    print(f"  Mean reward: {baseline_train['mean_reward']:.4f}")
    print(f"  Mean height: {baseline_train['mean_height']:.4f}m")

    # Reset for trained model test
    vec1 = DummyVecEnv([make_env_train])
    vn1 = VecNormalize.load(str(norm_path), vec1)
    vn1.training = False
    vn1.norm_reward = False

    print("\nTrained model:")
    trained_train = test_policy(model, vn1, steps=500, label="Trained")
    print(f"  Distance: {trained_train['final_position']:.4f}m")
    print(f"  Mean velocity: {trained_train['mean_fwd_vel']:.4f} m/s")
    print(f"  Mean reward: {trained_train['mean_reward']:.4f}")
    print(f"  Mean height: {trained_train['mean_height']:.4f}m")
    print(f"  Mean action magnitude: {trained_train['mean_action_mag']:.4f}")

    # Test 2: Evaluation gait (mismatched)
    print("\n[Test 2] Evaluation Gait (phase4_viewer gait params)")
    print("-"*70)

    gait_eval = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )

    def make_env_eval():
        return ResidualWalkEnv(model_path="model/world_train.xml", gait_params=gait_eval)

    vec2 = DummyVecEnv([make_env_eval])
    vn2 = VecNormalize.load(str(norm_path), vec2)
    vn2.training = False
    vn2.norm_reward = False

    print("Baseline (zero residuals):")
    baseline_eval = test_policy(None, vn2, steps=500, label="Baseline")
    print(f"  Distance: {baseline_eval['final_position']:.4f}m")
    print(f"  Mean velocity: {baseline_eval['mean_fwd_vel']:.4f} m/s")
    print(f"  Mean reward: {baseline_eval['mean_reward']:.4f}")

    vec2 = DummyVecEnv([make_env_eval])
    vn2 = VecNormalize.load(str(norm_path), vec2)
    vn2.training = False
    vn2.norm_reward = False

    print("\nTrained model:")
    trained_eval = test_policy(model, vn2, steps=500, label="Trained")
    print(f"  Distance: {trained_eval['final_position']:.4f}m")
    print(f"  Mean velocity: {trained_eval['mean_fwd_vel']:.4f} m/s")
    print(f"  Mean reward: {trained_eval['mean_reward']:.4f}")
    print(f"  Mean action magnitude: {trained_eval['mean_action_mag']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    train_diff = trained_train['mean_fwd_vel'] - baseline_train['mean_fwd_vel']
    eval_diff = trained_eval['mean_fwd_vel'] - baseline_eval['mean_fwd_vel']

    print(f"\nVelocity difference (trained - baseline):")
    print(f"  Training gait: {train_diff:+.4f} m/s")
    print(f"  Eval gait:     {eval_diff:+.4f} m/s")

    if train_diff < -0.05:
        print("\n⚠ WARNING: Trained model is SLOWER than baseline with training gait!")
        print("  → The model learned to minimize residuals, not maximize velocity")
        print("  → Reward function needs rebalancing (see DIAGNOSIS.md)")

    if trained_train['mean_action_mag'] < 0.1:
        print("\n⚠ WARNING: Model uses very small residuals (mean < 0.1)")
        print("  → The model learned conservative behavior")
        print("  → It's barely modifying the baseline controller")

    if abs(eval_diff - train_diff) > 0.05:
        print("\n⚠ WARNING: Performance differs significantly between gaits")
        print("  → Train/eval gait mismatch detected")
        print("  → Fix evaluation scripts to match training gait")

    print("\nSee DIAGNOSIS.md for detailed analysis and solutions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
