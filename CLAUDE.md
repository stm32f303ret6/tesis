# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quadruped robot simulation using MuJoCo physics engine. The robot has 4 legs, each using a parallel SCARA (Selective Compliance Assembly Robot Arm) mechanism with 3 degrees of freedom: tilt + two shoulder joints. The project includes inverse kinematics solvers and demonstration scripts.

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -U mujoco numpy

# Optional development tools
pip install black ruff mypy pytest

# For gait visualization (test scripts)
pip install matplotlib pillow
```

**Graphics requirement**: MuJoCo viewer requires OpenGL. For headless servers, set `export MUJOCO_GL=egl` before running.

## Running Demos

**IMPORTANT**: Always run scripts from the repository root so relative paths (`model/world.xml`) resolve correctly.

```bash
# Basic height control with sinusoidal motion
python3 height_control.py

# Smooth startup with eased tilt transition followed by height control
python3 smooth_startup.py

# Test IK solvers directly
python3 ik.py
```

## Development Commands

```bash
# Format code
black .

# Lint
ruff .

# Type check
mypy ik.py

# Run tests
python -m pytest test/ -v

# Run specific visualization tests
python3 test/test_bezier_gait.py
python3 test/explain_bezier_params.py
```

## Architecture

### Core IK Module (`ik.py`)

Contains all inverse kinematics solvers - keep this pure math with no I/O:

- `solve_2link_ik()`: Basic 2-link planar arm IK with elbow-up/down configuration
- `parallel_scara_ik()`: 2DOF parallel SCARA solver with 4 working modes:
  - Mode 1: Arm A elbow up, B down
  - Mode 2: Arm A elbow down, B up (default)
  - Mode 3: Both elbows up
  - Mode 4: Both elbows down
- `parallel_scara_ik_3dof()`: 3DOF version adding tilt control for lateral displacement
- `solve_leg_ik_3dof()`: Convenience wrapper for robot-specific leg IK

**Default leg parameters**:
- L1 (upper link): 0.045m (45mm)
- L2 (lower link): 0.06m (60mm)
- base_dist (parallel arm spacing): 0.021m (21mm)

### Demo Scripts

- `height_control.py`: Continuous sinusoidal height + lateral motion using 3DOF IK
- `smooth_startup.py`: Two-phase demo:
  1. 5-second camera setup delay
  2. Cubic-eased tilt transition to 0° over 2 seconds
  3. Transitions to height control movements

Both demos track the robot body with the camera automatically.

### Motor Control Indexing

The robot has 12 motors (4 legs × 3 DOF):
- Each leg: `[shoulder_L, shoulder_R, tilt]`
- Indices: RL(0-2), RL(3-5), FL(6-8), FR(9-11)
- Tilt motors specifically: `[2, 5, 8, 11]`

**Leg symmetry**: Front and rear legs use mirrored angles:
- Front legs: `ang1L`, `ang1R + π`, `tilt`
- Rear legs: `-ang1L`, `-ang1R - π`, `tilt`

### MuJoCo Models

- `model/world.xml`: Top-level scene with lighting, ground plane, includes robot.xml
- `model/robot.xml`: Robot definition with 4 parallel SCARA legs, each with tilt joint + dual shoulder mechanism
- `model/assets/*.stl`: 3D meshes for visualization
- `model/openscad/*.scad`: Source CAD files for generating STL meshes

## Coding Conventions

- Python 3, PEP 8, 4-space indentation
- snake_case for modules, functions, variables
- UPPER_CASE for constants
- Type hints for public functions in `ik.py`
- Keep IK functions pure (no I/O, no side effects)
- Keep viewer/demo logic in scripts, math/IK in `ik.py`

## Testing

- Framework: pytest
- Place tests in `test/` directory (note: currently `test/` not `tests/`) with `test_*.py` naming
- Focus on numeric assertions: reachability checks, angle ranges, forward kinematics validation
- Target >80% coverage of core IK logic

### Running Tests

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test file
python3 test/test_bezier_gait.py

# Generate gait visualization (creates gait_trajectory_static.png and gait_animation.gif)
python3 test/explain_bezier_params.py
```

### Test Coverage

- `test_bezier_gait.py`: Bezier gait trajectory generation, FK/IK consistency verification, and animation
  - Includes forward kinematics (2DOF and 3DOF) for verification
  - Generates visualizations of gait cycles
  - Validates that FK(IK(target)) ≈ target
- `explain_bezier_params.py`: Parameter visualization tool for understanding gait parameters

## Commit Style

Use Conventional Commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code restructuring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## Important Notes

- Do not break relative paths to `model/*` - many scripts depend on running from repo root
- When changing IK defaults, update usage examples in `ik.py` and verify demos still run
- Keep STL/SCAD file paths stable; avoid committing large binaries beyond existing assets
- The robot body has a free joint (6DOF floating base) - it can move in simulation

## Gait Development

The project includes Bezier-based gait trajectory tools in `test/`:
- **Gait parameters**: `step_length` (stride), `step_height` (clearance), `stance_height` (body height)
- **Safe ranges** (from `test/bezier_params_explanation.txt`):
  - stance_height: -0.04 to -0.09m (-40mm to -90mm)
  - step_length: 0.02 to 0.06m (20mm to 60mm)
  - step_height: 0.01 to 0.05m (10mm to 50mm)
- **Gait cycle**: 50% swing phase (Bezier curve), 50% stance phase (linear motion)
- **Forward kinematics**: Available in `test/test_bezier_gait.py` for FK/IK verification (validates <5mm error)
- Generate visualizations with `python3 test/explain_bezier_params.py`
