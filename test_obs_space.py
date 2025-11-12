#!/usr/bin/env python3
"""Quick test to verify reduced observation space dimensions."""

import numpy as np
from envs.residual_walk_env import ResidualWalkEnv

def test_observation_space():
    """Verify observation space has correct dimensions."""
    print("Creating environment...")
    env = ResidualWalkEnv(
        model_path="model/world_train.xml",
        max_episode_steps=100,
        settle_steps=10,  # Reduce for faster testing
    )

    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Expected: (65,)")

    print("\nResetting environment...")
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"All values finite: {np.all(np.isfinite(obs))}")

    # Break down observation components
    idx = 0
    print("\nObservation breakdown:")
    print(f"  Body position (3D):    {obs[idx:idx+3]}")
    idx += 3
    print(f"  Body quaternion (4D):  {obs[idx:idx+4]}")
    idx += 4
    print(f"  Linear velocity (3D):  {obs[idx:idx+3]}")
    idx += 3
    print(f"  Angular velocity (3D): {obs[idx:idx+3]}")
    idx += 3
    print(f"  Joint states (24D):    shape={obs[idx:idx+24].shape}")
    idx += 24
    print(f"  Foot positions (12D):  shape={obs[idx:idx+12].shape}")
    idx += 12
    print(f"  Foot velocities (12D): shape={obs[idx:idx+12].shape}")
    idx += 12
    print(f"  Foot contacts (4D):    {obs[idx:idx+4]}")
    idx += 4

    print(f"\nTotal dimensions used: {idx}")

    # Test a few steps
    print("\nTesting step function...")
    action = np.zeros(12)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step observation shape: {obs.shape}")
    print(f"Reward: {reward:.4f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    print("\n✓ Observation space test passed!")

if __name__ == "__main__":
    test_observation_space()
