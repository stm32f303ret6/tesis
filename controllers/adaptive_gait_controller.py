"""Adaptive gait controller with trainable parameters.

This controller extends BezierGaitResidualController to support online
adaptation of gait parameters. The policy outputs both:
1. Gait parameter deltas (high-level adaptation)
2. Per-leg residual corrections (low-level adjustments)

This approach enables the robot to adapt its gait to rough terrain by
learning to modulate step height, step length, cycle time, and body height.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from gait_controller import DiagonalGaitController, GaitParameters, LEG_NAMES


class AdaptiveGaitController:
    """Gait controller with online parameter adaptation.

    Args:
        base_params: Nominal gait parameters (used as defaults)
        residual_scale: Max absolute residual per axis [meters]
        param_ranges: Min/max bounds for each trainable parameter
    """

    # Default parameter ranges (min, max, default)
    DEFAULT_RANGES = {
        "step_height": (0.015, 0.06, 0.04),
        "step_length": (0.03, 0.08, 0.06),
        "cycle_time": (0.6, 1.2, 0.8),
        "body_height": (0.04, 0.08, 0.05),
    }

    def __init__(
        self,
        base_params: Optional[GaitParameters] = None,
        residual_scale: float = 0.01,
        param_ranges: Optional[Dict[str, tuple]] = None,
    ) -> None:
        # Store base parameters as reference
        self.base_params = base_params or GaitParameters()
        self.residual_scale = float(residual_scale)

        # Parameter ranges for clipping
        self.param_ranges = param_ranges or self.DEFAULT_RANGES.copy()

        # Current adapted parameters (start at base)
        self.current_params = GaitParameters(
            body_height=self.base_params.body_height,
            step_length=self.base_params.step_length,
            step_height=self.base_params.step_height,
            cycle_time=self.base_params.cycle_time,
            swing_shape=self.base_params.swing_shape,
            lateral_offsets=self.base_params.lateral_offsets.copy(),
        )

        # Underlying gait controller (will be recreated when params change)
        self.base_controller = DiagonalGaitController(self.current_params)

        # Track when we need to rebuild the controller
        self._params_dirty = False

    def reset(self) -> None:
        """Reset to base parameters and phase."""
        # Reset to original base parameters
        self.current_params = GaitParameters(
            body_height=self.base_params.body_height,
            step_length=self.base_params.step_length,
            step_height=self.base_params.step_height,
            cycle_time=self.base_params.cycle_time,
            swing_shape=self.base_params.swing_shape,
            lateral_offsets=self.base_params.lateral_offsets.copy(),
        )

        # Rebuild controller with reset parameters
        self.base_controller = DiagonalGaitController(self.current_params)
        self._params_dirty = False

    def update_parameters(self, param_deltas: Dict[str, float]) -> None:
        """Apply deltas to gait parameters with clipping.

        Args:
            param_deltas: Dict with keys like "step_height", "step_length", etc.
                         Values are deltas to add to current parameters.
        """
        # Apply deltas and clip to valid ranges
        for param_name, delta in param_deltas.items():
            if param_name not in self.param_ranges:
                continue

            min_val, max_val, _ = self.param_ranges[param_name]
            current_val = getattr(self.current_params, param_name)
            new_val = float(np.clip(current_val + delta, min_val, max_val))
            setattr(self.current_params, param_name, new_val)

        # Mark that we need to rebuild the controller
        self._params_dirty = True

    def _rebuild_controller_if_needed(self) -> None:
        """Rebuild the base controller if parameters changed.

        Note: This recreates the controller, which resets the gait phase.
        For smoother transitions, consider implementing parameter interpolation.
        """
        if self._params_dirty:
            # Save current phase to restore after rebuild
            old_phase = self.base_controller.phase_elapsed
            old_state = getattr(self.base_controller, "state", "pair_a_swing")

            # Rebuild with new parameters
            self.base_controller = DiagonalGaitController(self.current_params)

            # Restore phase (approximately - state duration may have changed)
            self.base_controller.phase_elapsed = old_phase
            # Restore state
            if old_state == "pair_b_swing":
                self.base_controller.to_pair_b_swing()  # type: ignore

            self._params_dirty = False

    def update_with_residuals(
        self,
        dt: float,
        residuals: Dict[str, np.ndarray],
        param_deltas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Update gait with parameter adaptation and residuals.

        Args:
            dt: Timestep [seconds]
            residuals: Per-leg residual offsets {"FL": [dx, dy, dz], ...}
            param_deltas: Optional parameter deltas to apply this step

        Returns:
            Final foot targets per leg
        """
        # Apply parameter updates if provided
        if param_deltas is not None:
            self.update_parameters(param_deltas)

        # Rebuild controller if parameters changed
        self._rebuild_controller_if_needed()

        # Get nominal targets from base controller
        base_targets = self.base_controller.update(dt)

        # Add clipped residuals
        final_targets: Dict[str, np.ndarray] = {}
        for leg in LEG_NAMES:
            base = np.asarray(base_targets[leg], dtype=float)
            res = np.asarray(residuals.get(leg, np.zeros(3)), dtype=float)
            if res.shape != (3,):
                raise ValueError(f"Residual for leg {leg} must be shape (3,), got {res.shape}")
            res_clipped = np.clip(res, -self.residual_scale, self.residual_scale)
            final_targets[leg] = base + res_clipped

        return final_targets

    def get_current_parameters(self) -> Dict[str, float]:
        """Return current adapted parameter values."""
        return {
            "step_height": self.current_params.step_height,
            "step_length": self.current_params.step_length,
            "cycle_time": self.current_params.cycle_time,
            "body_height": self.current_params.body_height,
        }

    def get_phase_info(self) -> Dict[str, float]:
        """Return gait phase information (see BezierGaitResidualController)."""
        phase_elapsed = float(self.base_controller.phase_elapsed)
        state_duration = float(self.base_controller.state_duration)
        phase_norm = 0.0 if state_duration <= 0 else phase_elapsed / state_duration
        active_pair = 0 if getattr(self.base_controller, "state", "pair_a_swing") == "pair_a_swing" else 1
        return {
            "phase_elapsed": phase_elapsed,
            "state_duration": state_duration,
            "phase_normalized": phase_norm,
            "active_pair": float(active_pair),
        }

    def get_swing_stance_flags(self) -> Dict[str, int]:
        """Return 1 for swing legs, 0 for stance legs."""
        flags: Dict[str, int] = {}
        active_swing = set(self.base_controller.active_swing_pair)
        for leg in LEG_NAMES:
            flags[leg] = 1 if leg in active_swing else 0
        return flags
