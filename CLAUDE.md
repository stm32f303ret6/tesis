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

### Simulation Control (`height_control.py`)
The main demo script orchestrating MuJoCo simulation:
- Loads robot model from `model/world.xml`
- `set_leg_height(x, y, z)`: Sets all four legs to target position using 3DOF IK
- Implements sinusoidal height oscillation with optional lateral motion
- Camera follows robot body during simulation
- **Control mapping:** Front legs use indices 6-11, rear legs 0-5; rear shoulders are mirrored (negated)

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
pip install mujoco numpy
```

### Running Simulations
```bash
# Main height control demo (interactive viewer)
python3 height_control.py

# Headless mode (for CI/testing)
MUJOCO_GL=egl python3 height_control.py

# IK verification tests
python3 ik.py
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

1. **Coordinate frame confusion**: The IK operates in leg-local frame where Z points down (gravity direction). Target positions should be negative Z for downward reach.

2. **Control index mapping**: Front legs (FL, FR) use ctrl indices 6-11, rear legs (RL, RR) use 0-5. Each leg has 3 actuators: left shoulder, right shoulder (+π offset), tilt.

3. **Rear leg mirroring**: Rear shoulder angles must be negated to maintain symmetric gait.

4. **Asset regeneration**: When updating SCAD files, remember to regenerate STL and copy to `model/assets/` before testing.

5. **Reachability limits**: Max reach = L1 + L2 = 0.105m. Current code uses 15-90% of max reach as safe working range.

## Commit Style

Match repository history with brief, present-tense subjects (e.g., "adding primitives model", "changing diagonal_gait"). Reference issues or design docs in body when relevant. Note MuJoCo version or asset changes in commit messages.
