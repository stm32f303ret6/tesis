"""Test script for Walk2 environment."""

import jax
import jax.numpy as jnp
import walk2_env


def test_environment():
    """Test Walk2 environment with random actions."""
    print("="*60)
    print("Testing Walk2 Environment")
    print("="*60)

    # Load environment (use flat terrain for now since hfield.png is missing)
    print("\n1. Loading environment...")
    env = walk2_env.load('Walk2JoystickFlat')  # Changed from Rough to Flat
    print(f"   ✓ Environment loaded")
    print(f"   Action size: {env.action_size}")
    print(f"   Observation size: {env.observation_size}")

    # Test reset
    print("\n2. Testing reset...")
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    print(f"   ✓ Reset successful")
    print(f"   Observation shape: {state.obs.shape}")
    print(f"   Reward: {state.reward}")
    print(f"   Done: {state.done}")
    print(f"   Command: {state.info['command']}")

    # Test multiple steps with random actions
    print("\n3. Testing steps with random actions...")
    num_steps = 100
    rng = jax.random.PRNGKey(0)

    for step in range(num_steps):
        # Sample random action
        rng, action_key = jax.random.split(rng)
        action = jax.random.uniform(
            action_key,
            shape=(env.action_size,),
            minval=-1.0,
            maxval=1.0
        )

        # Execute step
        state = env.step(state, action)

        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"   Step {step+1}/{num_steps}:")
            print(f"     Reward: {state.reward:.4f}")
            print(f"     Height: {state.metrics['body_height']:.4f}")
            print(f"     Speed: {state.metrics['speed']:.4f}")
            print(f"     Done: {state.done}")

        # Check if episode ended
        if state.done:
            print(f"   Episode terminated at step {step+1}")
            # Reset for testing
            rng, reset_key = jax.random.split(rng)
            state = env.reset(reset_key)
            print(f"   Environment reset, new command: {state.info['command']}")

    print(f"\n   ✓ Completed {num_steps} steps successfully")

    # Test JIT compilation
    print("\n4. Testing JAX JIT compilation...")

    @jax.jit
    def jit_step(state, action):
        return env.step(state, action)

    @jax.jit
    def jit_reset(rng):
        return env.reset(rng)

    # Test JIT reset
    rng = jax.random.PRNGKey(123)
    state = jit_reset(rng)
    print(f"   ✓ JIT reset successful")

    # Test JIT step
    action = jnp.zeros(env.action_size)
    state = jit_step(state, action)
    print(f"   ✓ JIT step successful")

    # Test multiple JIT steps (should be fast)
    print(f"   Running 1000 JIT-compiled steps...")
    import time
    start = time.time()
    for _ in range(1000):
        rng, action_key = jax.random.split(rng)
        action = jax.random.uniform(
            action_key,
            shape=(env.action_size,),
            minval=-1.0,
            maxval=1.0
        )
        state = jit_step(state, action)
        if state.done:
            rng, reset_key = jax.random.split(rng)
            state = jit_reset(reset_key)
    elapsed = time.time() - start
    print(f"   ✓ 1000 steps in {elapsed:.2f}s ({1000/elapsed:.0f} steps/sec)")

    # Summary
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nEnvironment summary:")
    print(f"  Action space: {env.action_size}D continuous (normalized to [-1, 1])")
    print(f"  Observation space: {env.observation_size}D")
    print(f"    - Joint positions relative to home (12)")
    print(f"    - Joint velocities (12)")
    print(f"    - Gravity vector in body frame (3)")
    print(f"    - Angular velocity (3)")
    print(f"    - Last action (12)")
    print(f"    - Velocity command (3)")
    print(f"  Control frequency: {1/env._config.control_dt:.0f} Hz")
    print(f"  Simulation frequency: {1/env._config.sim_dt:.0f} Hz")
    print(f"  Physics substeps: {env._n_substeps}")

    return env, state


if __name__ == '__main__':
    env, final_state = test_environment()
