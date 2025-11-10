"""Residual wrapper for the diagonal Bézier gait controller.

This module augments the existing DiagonalGaitController by enabling per-leg
residual footpoint offsets that are added to the nominal Bézier targets.

Frames/units:
- Foot targets and residuals are in the controller's leg-local frame [meters].
- Residuals are clipped to +/- ``residual_scale`` to keep IK stable by default.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gait_controller import DiagonalGaitController, GaitParameters, LEG_NAMES


class BezierGaitResidualController:
    """Wraps DiagonalGaitController with residual support.

    Args:
        params: Gait parameter bundle for the underlying controller.
        residual_scale: Max absolute residual per axis [meters].
    """

    def __init__(
        self,
        params: Optional[GaitParameters] = None,
        residual_scale: float = 0.02,
    ) -> None:
        self.base_controller = DiagonalGaitController(params)
        self.residual_scale = float(residual_scale)

    def reset(self) -> None:
        """Reset the underlying controller phase and targets."""
        self.base_controller.reset()

    def update_with_residuals(
        self,
        dt: float,
        residuals: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Get nominal Bézier targets and add clipped residuals.

        Args:
            dt: Timestep [seconds]. If ``<= 0``, phase is not advanced.
            residuals: Mapping ``{"FL": np.array([dx, dy, dz]), ...}``.

        Returns:
            Dict of final foot targets per leg in the same leg-local frame.
        """
        # Nominal targets from the base controller
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

    def get_phase_info(self) -> Dict[str, float]:
        """Return gait phase information useful for observations.

        Keys:
            - ``phase_elapsed`` [s]
            - ``state_duration`` [s]
            - ``phase_normalized`` in [0, 1]
            - ``active_pair`` (0 for pair_a_swing, 1 for pair_b_swing)
        """
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

