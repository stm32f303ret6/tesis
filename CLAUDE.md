# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quadruped robot simulation project using MuJoCo physics engine. The robot features a novel 3DOF parallel SCARA leg design with tilt control, allowing independent control of leg height, lateral position, and tilt angle.

## Core Architecture

### Inverse Kinematics System (`ik.py`)
The IK module provides the mathematical foundation for leg control:
- `solve_2link_ik()`: Basic 2-link planar IK solver using law of cosines
- `parallel_scara_ik()`: 2DOF parallel SCARA mechanism with 4 working modes (different elbow configurations)
- `parallel_scara_ik_3dof()`: Full 3DOF IK including tilt control for 3D target positions
- `solve_leg_ik_3dof()`: Convenience wrapper with robot-specific default parameters

**Key parameters:**
- `L1 = 0.045` m (upper link length)
- `L2 = 0.06` m (lower link length)
- `base_dist = 0.021` m (distance between parallel arms)
- `mode`: 1=A up/B down, 2=A down/B up (default), 3=both up, 4=both down

The 3DOF IK transforms 3D targets (x, y, z) into joint angles (tilt, shoulder_L, shoulder_R) by:
1. Computing or accepting a tilt angle to handle y-displacement
2. Projecting the target into the tilted leg's planar frame
3. Solving the 2DOF SCARA IK in that plane

### Gait Controller (`gait_controller.py`)
State machine-based diagonal gait generator using `transitions` library for coordination and `bezier` for smooth swing trajectories:
- `DiagonalGaitController`: Main controller class managing trot gait with diagonal leg pairs
- `GaitParameters`: Dataclass for gait configuration (body_height, step_length, step_height, cycle_time, swing_shape, lateral_offsets)
- **Diagonal pairs:** FL+RR (pair_a), FR+RL (pair_b) alternate between swing and stance
- **State machine:** Two states (`pair_a_swing`, `pair_b_swing`) with automatic transitions every half-cycle
- **Swing trajectory:** Cubic Bézier curve with configurable shape parameter for lift/touchdown dynamics
- **Stance trajectory:** Linear sweep from front to rear over the stance duration

The controller outputs per-leg foot targets in leg-local frame, which are then transformed and fed to IK.

### Main Simulation Script (`height_control.py`)
Primary entry point for running the robot simulation:
- Loads robot model from `model/world_train.xml` (rough terrain) or `model/world.xml` (flat)
- `LEG_CONTROL` dictionary: Maps leg names to actuator indices, signs, and offsets
- `apply_leg_angles()`: Transforms IK output (tilt, shoulder_L, shoulder_R) into MuJoCo actuator commands
- `apply_gait_targets()`: Main control loop that evaluates gait, solves IK, applies commands
- `FORWARD_SIGN = -1.0`: Flips controller +X to match leg IK frame for forward motion
- **Control mapping:** FL=(0,1,2), RL=(3,4,5), FR=(6,7,8), RR=(9,10,11); rear shoulders negated, front/right shoulders offset by π

### MuJoCo Model Structure
- `model/world.xml`: Flat ground plane scene with checker texture
- `model/world_train.xml`: Rough terrain scene using heightfield from `hfield.png` for training/testing
- `model/robot.xml`: Complete robot definition with 4 legs, each having:
  - Tilt joint (1 DOF, axis=[1,0,0])
  - Two shoulder joints (parallel SCARA, axis=[0,-1,0])
  - Two elbow joints (passive/constrained)
- `model/assets/`: STL mesh files for all robot parts
- `model/openscad/`: Editable CAD sources (`.scad`) and generated `.stl` files
- `model/primitives_model/`: Simplified box geometry model for debugging

**Important:** Joint names and body names in XML must stay stable to avoid breaking control code that references them by name.

### Utility Scripts
- `foot_range_calculator.py`: Calculates reachable workspace for each foot using IK, validates safe parameter ranges
- `tests/compare_world_trajectories.py`: Compares robot trajectories between flat and rough terrain, generates matplotlib plots

## Development Commands

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mujoco numpy transitions bezier matplotlib
```

### Running Simulations
```bash
# Main gait controller demo on rough terrain (default)
python3 height_control.py

# Run on flat terrain (edit height_control.py to load model/world.xml)
python3 height_control.py

# Headless mode (for CI/testing, requires MUJOCO_GL=egl)
MUJOCO_GL=egl timeout 60 python3 height_control.py

# IK verification tests
python3 ik.py

# Calculate foot workspace and validate safe ranges
python3 foot_range_calculator.py

# Compare trajectories between flat and rough terrain
python3 tests/compare_world_trajectories.py
```

### Asset Pipeline
When modifying robot geometry:
1. Edit `.scad` files in `model/openscad/`
2. Regenerate STL: `openscad -o model/assets/<part>.stl model/openscad/<part>.scad`
3. Verify mesh orientation and scale in MuJoCo before committing
4. Update link lengths in code if dimensions changed

## Coding Conventions

### Python Style
- Follow PEP 8: 4-space indentation, line length ≤ 100
- Module constants in UPPER_SNAKE_CASE, functions/variables in snake_case
- Group imports: standard library, third-party (numpy, mujoco), local modules
- Keep docstrings focused on coordinate frames and units (meters, radians)

### Testing Approach
- `ik.py` contains built-in verification when run as main script
- For new features, add pytest-compatible tests in `tests/test_*.py`
- Before committing kinematic changes, visually verify with `python3 height_control.py`
- Check console for MuJoCo warnings about joint limits or contacts

## Common Pitfalls

1. **Coordinate frame confusion**: The IK operates in leg-local frame where Z points down (gravity direction). Target positions should be negative Z for downward reach. The gait controller uses forward +X, but this is flipped via `FORWARD_SIGN = -1.0` to match leg IK frame.

2. **Control index mapping**: FL=(0,1,2), RL=(3,4,5), FR=(6,7,8), RR=(9,10,11). Each leg has 3 actuators in order: left shoulder, right shoulder, tilt. **Not** front=(6-11) rear=(0-5) as previously documented.

3. **Shoulder angle transformations**: Rear legs (FL, RL) have shoulders negated (`sign=-1.0`), front/right legs (FR, RR) have right shoulder offset by π. See `LEG_CONTROL` dictionary in height_control.py:24-29.

4. **Asset regeneration**: When updating SCAD files, remember to regenerate STL and copy to `model/assets/` before testing.

5. **Reachability limits**: Max reach = L1 + L2 = 0.105m. Practical working range validated by `foot_range_calculator.py`.

6. **Gait parameters**: When tuning `GaitParameters`, ensure step_height + body_height doesn't exceed reachable workspace. Use foot_range_calculator.py to verify targets are reachable before testing in simulation.

7. **State machine timing**: The gait controller auto-transitions every `state_duration = cycle_time / 2.0`. Ensure MuJoCo timestep is small enough to capture smooth trajectory updates (default 0.002s works well).

## Commit Style

Match repository history with brief, present-tense subjects (e.g., "adding primitives model", "changing diagonal_gait"). Reference issues or design docs in body when relevant. Note MuJoCo version or asset changes in commit messages.
