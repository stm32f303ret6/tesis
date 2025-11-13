#!/usr/bin/env python3
"""Unit tests and smoke tests for AdaptiveGaitController.

Run this script to verify the adaptive controller works correctly before training.

Usage:
    python3 test_adaptive_controller.py
"""

from __future__ import annotations
import numpy as np

from controllers.adaptive_gait_controller import AdaptiveGaitController
from gait_controller import GaitParameters, LEG_NAMES


def test_initialization():
    """Test that controller initializes correctly."""
    print("Testing initialization...")

    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )

    controller = AdaptiveGaitController(
        base_params=base_params,
        residual_scale=0.01,
    )

    # Check that current params match base params initially
    current = controller.get_current_parameters()
    assert abs(current["body_height"] - 0.05) < 1e-6
    assert abs(current["step_length"] - 0.06) < 1e-6
    assert abs(current["step_height"] - 0.04) < 1e-6
    assert abs(current["cycle_time"] - 0.8) < 1e-6

    print("  ✓ Initialization correct")


def test_parameter_update():
    """Test parameter updates with clipping."""
    print("Testing parameter updates...")

    # Use explicit base parameters
    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )
    controller = AdaptiveGaitController(base_params=base_params)

    # Apply valid deltas
    controller.update_parameters({
        "step_height": 0.01,   # 0.04 + 0.01 = 0.05
        "step_length": -0.01,  # 0.06 - 0.01 = 0.05
    })

    current = controller.get_current_parameters()
    assert abs(current["step_height"] - 0.05) < 1e-6
    assert abs(current["step_length"] - 0.05) < 1e-6

    # Test clipping (try to go below minimum)
    controller.update_parameters({
        "step_height": -1.0,  # Should clip to min (0.015)
    })

    current = controller.get_current_parameters()
    assert abs(current["step_height"] - 0.015) < 1e-6

    # Test clipping (try to go above maximum)
    controller.update_parameters({
        "step_length": 1.0,  # Should clip to max (0.08)
    })

    current = controller.get_current_parameters()
    assert abs(current["step_length"] - 0.08) < 1e-6

    print("  ✓ Parameter updates and clipping correct")


def test_residual_application():
    """Test that residuals are applied correctly."""
    print("Testing residual application...")

    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )
    controller = AdaptiveGaitController(base_params=base_params, residual_scale=0.02)

    # Apply zero residuals - should get nominal targets
    zero_residuals = {leg: np.zeros(3) for leg in LEG_NAMES}
    targets1 = controller.update_with_residuals(0.0, zero_residuals)

    # Apply non-zero residuals
    residuals = {leg: np.array([0.01, 0.0, -0.01]) for leg in LEG_NAMES}
    targets2 = controller.update_with_residuals(0.0, residuals)

    # Check that residuals were added
    for leg in LEG_NAMES:
        diff = targets2[leg] - targets1[leg]
        assert abs(diff[0] - 0.01) < 1e-6, f"Leg {leg} x-residual not applied"
        assert abs(diff[2] - (-0.01)) < 1e-6, f"Leg {leg} z-residual not applied"

    print("  ✓ Residual application correct")


def test_combined_update():
    """Test updating both parameters and residuals together."""
    print("Testing combined parameter + residual update...")

    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )
    controller = AdaptiveGaitController(base_params=base_params, residual_scale=0.02)

    # Get baseline targets
    zero_res = {leg: np.zeros(3) for leg in LEG_NAMES}
    baseline = controller.update_with_residuals(0.0, zero_res)

    # Update with both param deltas and residuals
    param_deltas = {"step_height": 0.01}
    residuals = {leg: np.array([0.01, 0.0, 0.0]) for leg in LEG_NAMES}

    targets = controller.update_with_residuals(0.0, residuals, param_deltas)

    # Parameters should have changed
    current = controller.get_current_parameters()
    assert abs(current["step_height"] - 0.05) < 1e-6

    # Targets should reflect both parameter change AND residuals
    # (Hard to test exact values since parameter change rebuilds Bezier curve)
    for leg in LEG_NAMES:
        assert targets[leg].shape == (3,)
        assert np.all(np.isfinite(targets[leg]))

    print("  ✓ Combined update correct")


def test_phase_continuity():
    """Test that phase continues correctly after parameter updates."""
    print("Testing phase continuity...")

    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )
    controller = AdaptiveGaitController(base_params=base_params)
    zero_res = {leg: np.zeros(3) for leg in LEG_NAMES}

    # Advance phase
    for _ in range(10):
        controller.update_with_residuals(0.01, zero_res)

    phase_before = controller.get_phase_info()["phase_elapsed"]

    # Update parameters (triggers rebuild)
    param_deltas = {"step_height": 0.005}
    controller.update_with_residuals(0.01, zero_res, param_deltas)

    phase_after = controller.get_phase_info()["phase_elapsed"]

    # Phase should have advanced, not reset to zero
    assert phase_after > phase_before, "Phase was reset during parameter update"

    print("  ✓ Phase continuity preserved")


def test_reset():
    """Test that reset returns to base parameters."""
    print("Testing reset...")

    base_params = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8,
    )

    controller = AdaptiveGaitController(base_params=base_params)

    # Modify parameters
    controller.update_parameters({
        "step_height": 0.01,
        "step_length": -0.01,
        "cycle_time": 0.1,
    })

    current = controller.get_current_parameters()
    assert abs(current["step_height"] - 0.04) > 1e-6  # Should be different from base

    # Reset
    controller.reset()

    # Should return to base values
    current = controller.get_current_parameters()
    assert abs(current["step_height"] - 0.04) < 1e-6
    assert abs(current["step_length"] - 0.06) < 1e-6
    assert abs(current["cycle_time"] - 0.8) < 1e-6

    # Phase should be reset
    phase = controller.get_phase_info()["phase_elapsed"]
    assert abs(phase) < 1e-6

    print("  ✓ Reset correct")


def smoke_test_environment():
    """Smoke test the AdaptiveGaitEnv."""
    print("\nSmoke testing AdaptiveGaitEnv...")

    from envs.adaptive_gait_env import AdaptiveGaitEnv

    env = AdaptiveGaitEnv(
        model_path="model/world_train.xml",
        max_episode_steps=100,
        settle_steps=0,
    )

    print("  - Environment created")

    # Check spaces
    assert env.action_space.shape == (16,), f"Action space should be 16D, got {env.action_space.shape}"
    assert env.observation_space.shape[0] == 69, f"Obs space should be 69D, got {env.observation_space.shape}"
    print(f"  - Action space: {env.action_space.shape}")
    print(f"  - Observation space: {env.observation_space.shape}")

    # Reset
    obs, info = env.reset()
    assert obs.shape == (69,), f"Observation should be 69D, got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
    print("  - Reset successful")

    # Take random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (69,)
        assert np.isfinite(reward)
        assert "gait_params" in info

        if terminated or truncated:
            obs, info = env.reset()

    print("  - Random rollout successful")
    print("  ✓ Smoke test passed")


def main() -> int:
    print("=" * 80)
    print("Testing AdaptiveGaitController")
    print("=" * 80)
    print()

    try:
        test_initialization()
        test_parameter_update()
        test_residual_application()
        test_combined_update()
        test_phase_continuity()
        test_reset()
        smoke_test_environment()

        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        print("\nYou can now train with: python3 train_adaptive_gait_ppo.py")

        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
