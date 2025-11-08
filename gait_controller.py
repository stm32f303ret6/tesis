"""Diagonal gait generator using transitions and Bézier swing trajectories."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import bezier
import numpy as np
from transitions import Machine

try:
    from joystick_input import VelocityCommand
except ImportError:
    # Fallback definition if joystick_input not available
    @dataclass
    class VelocityCommand:
        vx: float = 0.0
        vy: float = 0.0
        omega: float = 0.0

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

    def __init__(self, params: Optional[GaitParameters] = None, robot_width: float = 0.1) -> None:
        self.params = params or GaitParameters()
        self.state_duration = self.params.cycle_time / 2.0
        self.phase_elapsed = 0.0
        self.active_swing_pair: Tuple[str, str] = DIAGONAL_PAIRS["pair_a"]
        self.active_stance_pair: Tuple[str, str] = DIAGONAL_PAIRS["pair_b"]

        # Velocity-based control
        self.current_velocity = VelocityCommand()
        self.robot_width = robot_width  # Distance between left and right legs (for rotation)

        # Step displacement based on current velocity
        self.step_x = 0.0
        self.step_y = 0.0
        self._leg_rotations: Dict[str, float] = {}  # Rotation-based step adjustments per leg

        self._update_step_parameters()
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

    def set_velocity_command(self, velocity: VelocityCommand) -> None:
        """Update the velocity command and recompute step parameters."""
        # Check if velocity changed significantly
        velocity_changed = (
            abs(velocity.vx - self.current_velocity.vx) > 0.01
            or abs(velocity.vy - self.current_velocity.vy) > 0.01
            or abs(velocity.omega - self.current_velocity.omega) > 0.05
        )

        if velocity_changed:
            self.current_velocity = velocity
            self._update_step_parameters()
            self._build_swing_curve()

    def _update_step_parameters(self) -> None:
        """Compute step displacements from current velocity command."""
        # Linear displacement per half-cycle (one stance phase)
        self.step_x = self.current_velocity.vx * self.state_duration

        # Lateral motion needs larger steps to overcome body sway
        # Apply gain factor to lateral steps for better translation
        # Tuned for balance between translation speed and stability
        lateral_gain = 2.2
        self.step_y = self.current_velocity.vy * self.state_duration * lateral_gain

        # Rotation: compute per-leg arc length based on distance from center
        # Positive omega = counter-clockwise rotation (left side forward, right side back)
        arc_displacement = self.current_velocity.omega * self.robot_width / 2.0 * self.state_duration

        # Leg-specific rotation adjustments (added to step_x)
        # FL, RL are on left side: add arc displacement
        # FR, RR are on right side: subtract arc displacement
        self._leg_rotations = {
            "FL": arc_displacement,
            "FR": -arc_displacement,
            "RL": arc_displacement,
            "RR": -arc_displacement,
        }

    def update(self, dt: float, velocity: Optional[VelocityCommand] = None) -> Dict[str, np.ndarray]:
        """Advance the gait by ``dt`` seconds and return per-leg foot targets."""
        # Update velocity if provided
        if velocity is not None:
            self.set_velocity_command(velocity)

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
        """Pre-compute a cubic Bézier curve for the swing phase based on current velocity."""
        # Use velocity-based step displacement (zero when stationary)
        half_step_x = self.step_x / 2.0
        half_step_y = self.step_y / 2.0

        lift_height = max(self.params.body_height - self.params.step_height, 1e-4)
        shape = np.clip(self.params.swing_shape, 0.05, 0.95)

        # Build curve with both X and Y displacement
        nodes = np.asfortranarray(
            [
                [-half_step_x, -half_step_x * shape, half_step_x * shape, half_step_x],
                [-half_step_y, -half_step_y * shape, half_step_y * shape, half_step_y],
                [self.params.body_height, lift_height, lift_height, self.params.body_height],
            ]
        )
        self.swing_curve = bezier.Curve(nodes, degree=3)

    def _evaluate_swing_curve(self, leg: str, tau: float) -> np.ndarray:
        """Evaluate the swing curve for a given leg with rotation adjustment."""
        tau_clamped = float(np.clip(tau, 0.0, 1.0))
        point = self.swing_curve.evaluate(tau_clamped).flatten()

        # Apply rotation adjustment to X position
        rotation_offset = self._leg_rotations.get(leg, 0.0)
        point[0] += rotation_offset

        return self._apply_lateral_offset(leg, point)

    def _evaluate_stance_path(self, leg: str, tau: float) -> np.ndarray:
        """Linearly sweep the stance foot from front to rear based on velocity."""
        tau_clamped = float(np.clip(tau, 0.0, 1.0))

        # Use velocity-based step displacement (zero when stationary)
        step_x = self.step_x
        step_y = self.step_y

        # Apply rotation adjustment
        rotation_offset = self._leg_rotations.get(leg, 0.0)

        # Stance sweeps through both X and Y
        # For continuous translation: stance legs move opposite to velocity direction
        # This keeps feet approximately stationary relative to ground while body translates
        half_step_x = step_x / 2.0
        half_step_y = step_y / 2.0

        # Stance sweeps from forward position to rear position
        # In body frame: feet appear to move backward/opposite-laterally as body translates forward/laterally
        x_pos = half_step_x - (step_x * tau_clamped) + rotation_offset
        y_pos = half_step_y - (step_y * tau_clamped)

        stance_point = np.array([x_pos, y_pos, self.params.body_height], dtype=float)
        return self._apply_lateral_offset(leg, stance_point)

    def _apply_lateral_offset(self, leg: str, point: np.ndarray) -> np.ndarray:
        """Shift the foot target by the configured lateral offset."""
        lateral = self.params.lateral_offsets.get(leg, 0.0)
        adjusted = point.copy()
        adjusted[1] += lateral  # Add offset to existing Y-component (don't replace)
        return adjusted
