#!/usr/bin/env python3
"""Test simplified reward function."""

import numpy as np
from envs.residual_walk_env import ResidualWalkEnv

def test_rewards():
    """Test reward computation with new simplified rewards."""
    print("Creating environment...")
    env = ResidualWalkEnv(
        model_path="model/world_train.xml",
        max_episode_steps=100,
        settle_steps=50,
    )

    print("\nResetting environment...")
    obs, info = env.reset()

    print("\nRunning 10 steps and checking rewards...")
    for step in range(10):
        action = np.zeros(12)  # Zero residuals
        obs, reward, terminated, truncated, info = env.step(action)

        reward_components = info.get("reward_components", {})
        print(f"\nStep {step + 1}:")
        print(f"  Total reward: {reward:.4f}")
        for key, value in reward_components.items():
            print(f"    {key}: {value:.4f}")

        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
            break

    print("\n✓ Reward test completed!")

if __name__ == "__main__":
    test_rewards()
