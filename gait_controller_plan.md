# Quadruped Gait Controller Plan

## Objectives
- Replace the passive height demo with a forward gait controller that keeps body pose stable while advancing the robot.
- Use the `transitions` state machine library to coordinate diagonal leg pairs (front-left & rear-right, front-right & rear-left).
- Generate smooth swing trajectories for each leg using 4-point Bézier curves from the `bezier` Python library.
- Keep IK and MuJoCo configurations consistent with existing link lengths and control ranges.

## Implementation Steps

1. **Review Existing Control & Kinematics**
   - Inspect `height_control.py` and `ik.py` to understand current leg control flow, joint order, and IK constraints.
   - Identify existing helper functions that can be reused for shoulder/tilt command mapping.
   - Document current indexing of actuators (`data.ctrl`) to avoid regressions.

2. **Introduce Dependencies**
   - Add `transitions` and `bezier` to the project setup instructions (and optionally a lightweight requirements file).
   - Verify imports under the current virtualenv instructions; capture any dependency-specific initialization (e.g., numpy compatibility).

3. **Design Gait Parameters**
   - Define canonical footfall timing (duty factor, swing/stance durations) and body velocity target.
   - Choose Bézier control points for swing trajectories (start stance point, lift-off, apex, touchdown) in leg frame coordinates.
   - Decide on stance path handling (linear interpolation vs. constant velocity) and how to blend with body translation.

4. **Implement Trajectory Generator**
   - Create a dedicated module (e.g., `gait_controller.py`) that:
     - Encapsulates gait configuration dataclasses (timing, step length, clearance).
     - Uses the `bezier` library to generate parameterized swing curves with evaluation helper functions.
     - Provides stance trajectory computation and foot pose targets for all four legs.
   - Ensure outputs match the IK input signature (`solve_leg_ik_3dof` expectations).

5. **Build State Machine with `transitions`**
   - Model states for each gait phase (e.g., `pair_a_swing`, `pair_b_swing`) ensuring diagonal pairing.
   - Configure transitions triggered by elapsed time or trajectory completion.
   - Hook state entry/exit callbacks to reset per-leg timers and select the proper trajectory segment.

6. **Integrate with MuJoCo Loop**
   - Replace the sinusoidal loop in `height_control.py` with the new controller:
     - Update the control loop to step the state machine, evaluate trajectories, solve IK, and push actuator commands.
     - Maintain camera tracking and rate control already present in the demo.
   - Add configurable parameters (speed, clearance) via CLI flags or module-level constants.

7. **Validation & Testing**
   - Extend `ik.py` spot-checks or create `tests/test_gait_controller.py` to cover Bézier path evaluation and state transitions.
   - Run `python height_control.py` (and optionally `MUJOCO_GL=egl python height_control.py`) to confirm stable forward motion without warnings.
   - Tune trajectory points and timings to avoid foot scuffing or body pitch oscillations.

## Deliverables
- New gait controller module with documentation and inline comments for non-obvious logic.
- Updated `height_control.py` (or renamed controller script) invoking the state machine-driven gait.
- Dependency updates and usage notes (README or project setup).
- Basic automated checks ensuring the controller evaluates without MuJoCo.
