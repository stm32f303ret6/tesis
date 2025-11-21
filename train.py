#!/usr/bin/env python3
"""Simple PPO training script using Stable-Baselines3."""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.quadruped_env import QuadrupedEnv


def main():
    # Create vectorized environment
    env = DummyVecEnv([lambda: QuadrupedEnv()])
    env = VecNormalize(env)

    # Train PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
    model.learn(total_timesteps=1_000_000)

    # Save
    model.save("quadruped_ppo")
    env.save("vec_normalize.pkl")


if __name__ == "__main__":
    main()
