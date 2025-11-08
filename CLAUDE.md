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
The locomotion planning module implementing diagonal trot gait:
- `DiagonalGaitController`: Finite-state machine for diagonal pair coordination using `transitions` library
- Two states: `pair_a_swing` (FL+RR swing) and `pair_b_swing` (FR+RL swing)
- Swing trajectories use cubic Bézier curves via `bezier` library for smooth foot motion
- Stance phase: linear sweep from front to back proportional to step length
- `GaitParameters`: Configurable gait settings (body_height, step_length, step_height, cycle_time, lateral_offsets)
- Returns per-leg 3D target positions `[x, y, z]` in leg-local frame

### Simulation Control (`height_control.py`)
The main integration script orchestrating MuJoCo simulation with gait control:
- Loads robot model from `model/world.xml`
- `apply_leg_angles()`: Maps IK output (tilt, shoulder_L, shoulder_R) to actuator control indices
- `apply_gait_targets()`: Queries gait controller, solves IK for each leg, applies to MuJoCo
- Camera follows robot body during simulation
- **Control mapping:** Each leg has 3 actuators at specific indices defined in `LEG_CONTROL` dict:
  - FL: indices (0,1,2), sign=-1, offset=-π
  - FR: indices (6,7,8), sign=1, offset=π
  - RL: indices (3,4,5), sign=-1, offset=-π
  - RR: indices (9,10,11), sign=1, offset=π

### MuJoCo Model Structure
- `model/world.xml`: Top-level scene with ground plane, lighting, and includes `robot.xml`
- `model/robot.xml`: Complete robot definition with 4 legs, each having:
  - Tilt joint (1 DOF, axis=[1,0,0])
  - Two shoulder joints (parallel SCARA, axis=[0,-1,0])
  - Two elbow joints (passive/constrained)
- `model/assets/`: STL mesh files for all robot parts
- `model/openscad/`: Editable CAD sources (`.scad`) and generated `.stl` files

**Important:** Joint names and body names in XML must stay stable to avoid breaking control code that references them by name.

## Development Commands

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running Simulations
```bash
# Main gait controller demo (interactive viewer with trot gait)
python3 height_control.py

# Headless mode (for CI/testing)
MUJOCO_GL=egl timeout 10 python3 height_control.py

# IK verification tests (standalone verification with test cases)
python3 ik.py

# Run unit tests
python3 -m pytest tests/
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
- `ik.py` contains built-in verification when run as main script (2DOF, 3DOF, height control tests)
- Unit tests in `tests/` use pytest (e.g., `test_gait_controller.py` verifies state transitions)
- Before committing kinematic/gait changes, visually verify with `python3 height_control.py`
- Check console for MuJoCo warnings about joint limits or contacts
- Headless validation: Use `MUJOCO_GL=egl timeout 10 python3 height_control.py` to verify no crashes

## Common Pitfalls

1. **Coordinate frame confusion**: The IK operates in leg-local frame where Z points down (gravity direction). Target positions should be negative Z for downward reach. Gait controller outputs are in this same frame.

2. **Control index mapping**: The `LEG_CONTROL` dict in `height_control.py` defines the exact mapping:
   - Each leg has 3 control indices: (left_shoulder, right_shoulder, tilt)
   - Front legs use different indices than rear legs
   - Sign and offset are applied per-leg to ensure symmetric motion

3. **Actuator sign conventions**: Left shoulders and tilts use raw angles, but right shoulders add an offset (±π) due to mechanical mirroring. The `apply_leg_angles()` function handles this automatically.

4. **Gait state timing**: `DiagonalGaitController.update()` expects real timestep `dt` in seconds. Using wrong timestep will desynchronize swing/stance phases.

5. **Asset regeneration**: When updating SCAD files, remember to regenerate STL and copy to `model/assets/` before testing.

6. **Reachability limits**: Max reach = L1 + L2 = 0.105m. Gait parameters should keep step_length and body_height within safe IK workspace to avoid None returns from `solve_leg_ik_3dof()`.

## Commit Style

Match repository history with brief, present-tense subjects (e.g., "adding primitives model", "changing diagonal_gait"). Reference issues or design docs in body when relevant. Note MuJoCo version or asset changes in commit messages.
