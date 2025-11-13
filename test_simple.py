"""Simple test for Walk2 environment."""

import jax
import jax.numpy as jnp
import walk2_env

print("Loading environment...")
env = walk2_env.load('Walk2JoystickFlat')
print(f"Environment loaded: action_size={env.action_size}, obs_size={env.observation_size}")

print("\nResetting environment...")
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
print(f"Reset complete: obs.shape={state.obs.shape}, reward={state.reward}, done={state.done}")

print("\nTaking one step...")
action = jnp.zeros(env.action_size)
state = env.step(state, action)
print(f"Step complete: reward={state.reward}, done={state.done}")

print("\n✓ Basic test passed!")
