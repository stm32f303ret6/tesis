"""Diagonal gait generator using transitions and Bézier swing trajectories."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import bezier
import numpy as np
from transitions import Machine

LEG_NAMES: Tuple[str, ...] = ("FL", "FR", "RL", "RR")
DIAGONAL_PAIRS: Dict[str, Tuple[str, str]] = {
    "pair_a": ("FL", "RR"),
    "pair_b": ("FR", "RL"),
}


@dataclass
class GaitParameters:
    """Configuration bundle for the trot gait."""

    body_height: float = 0.07
    step_length: float = 0.05
    step_height: float = 0.015
    cycle_time: float = 0.8
    swing_shape: float = 0.35
    lateral_offsets: Dict[str, float] = field(
        default_factory=lambda: {
            "FL": -0.0,
            "FR": 0.0,
            "RL": -0.0,
            "RR": 0.0,
        }
    )


class DiagonalGaitController:
    """Manages a trot gait with diagonal swing pairs."""

    states = ["pair_a_swing", "pair_b_swing"]

    def __init__(self, params: Optional[GaitParameters] = None) -> None:
        self.params = params or GaitParameters()
        self.state_duration = self.params.cycle_time / 2.0
        self.phase_elapsed = 0.0
        self.active_swing_pair: Tuple[str, str] = DIAGONAL_PAIRS["pair_a"]
        self.active_stance_pair: Tuple[str, str] = DIAGONAL_PAIRS["pair_b"]
        self._build_swing_curve()

        self.machine = Machine(
            model=self,
            states=self.states,
            initial="pair_a_swing",
            after_state_change="_update_active_pair",
            send_event=True,
        )
        self.machine.add_transition("toggle_pair", "pair_a_swing", "pair_b_swing")
        self.machine.add_transition("toggle_pair", "pair_b_swing", "pair_a_swing")
        self._update_active_pair(None)
        self._leg_targets: Dict[str, np.ndarray] = self._bootstrap_targets()

    def reset(self) -> None:
        """Reset the controller to its initial phase."""
        self.to_pair_a_swing()  # type: ignore[attr-defined]
        self.phase_elapsed = 0.0
        self._leg_targets = self._bootstrap_targets()

    def update(self, dt: float) -> Dict[str, np.ndarray]:
        """Advance the gait by ``dt`` seconds and return per-leg foot targets."""
        if dt <= 0.0:
            return self._leg_targets

        self.phase_elapsed += dt
        while self.phase_elapsed >= self.state_duration:
            self.phase_elapsed -= self.state_duration
            self.toggle_pair()

        phase = np.clip(self.phase_elapsed / self.state_duration, 0.0, 1.0)
        targets: Dict[str, np.ndarray] = {}

        for leg in self.active_swing_pair:
            targets[leg] = self._evaluate_swing_curve(leg, phase)

        for leg in self.active_stance_pair:
            targets[leg] = self._evaluate_stance_path(leg, phase)

        self._leg_targets = targets
        return targets

    def _bootstrap_targets(self) -> Dict[str, np.ndarray]:
        """Generate initial setpoints that match the current state."""
        phase = 0.0
        return {
            **{leg: self._evaluate_swing_curve(leg, phase) for leg in self.active_swing_pair},
            **{leg: self._evaluate_stance_path(leg, phase) for leg in self.active_stance_pair},
        }

    def _update_active_pair(self, event) -> None:  # type: ignore[override]
        """Refresh swing/stance assignments whenever the state machine toggles."""
        if self.state == "pair_a_swing":
            self.active_swing_pair = DIAGONAL_PAIRS["pair_a"]
            self.active_stance_pair = DIAGONAL_PAIRS["pair_b"]
        else:
            self.active_swing_pair = DIAGONAL_PAIRS["pair_b"]
            self.active_stance_pair = DIAGONAL_PAIRS["pair_a"]

    def _build_swing_curve(self) -> None:
        """Pre-compute a cubic Bézier curve for the swing phase."""
        half_step = self.params.step_length / 2.0
        lift_height = max(self.params.body_height - self.params.step_height, 1e-4)
        shape = np.clip(self.params.swing_shape, 0.05, 0.95)

        nodes = np.asfortranarray(
            [
                [-half_step, -half_step * shape, half_step * shape, half_step],
                [0.0, 0.0, 0.0, 0.0],
                [self.params.body_height, lift_height, lift_height, self.params.body_height],
            ]
        )
        self.swing_curve = bezier.Curve(nodes, degree=3)

    def _evaluate_swing_curve(self, leg: str, tau: float) -> np.ndarray:
        """Evaluate the swing curve for a given leg."""
        tau_clamped = float(np.clip(tau, 0.0, 1.0))
        point = self.swing_curve.evaluate(tau_clamped).flatten()
        return self._apply_lateral_offset(leg, point)

    def _evaluate_stance_path(self, leg: str, tau: float) -> np.ndarray:
        """Linearly sweep the stance foot from front to rear."""
        tau_clamped = float(np.clip(tau, 0.0, 1.0))
        half_step = self.params.step_length / 2.0
        x_pos = half_step - (self.params.step_length * tau_clamped)
        stance_point = np.array([x_pos, 0.0, self.params.body_height], dtype=float)
        return self._apply_lateral_offset(leg, stance_point)

    def _apply_lateral_offset(self, leg: str, point: np.ndarray) -> np.ndarray:
        """Shift the foot target by the configured lateral offset."""
        lateral = self.params.lateral_offsets.get(leg, 0.0)
        adjusted = point.copy()
        adjusted[1] = lateral
        return adjusted
