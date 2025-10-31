# Repository Guidelines

## Project Structure & Module Organization
- `height_control.py` is the live MuJoCo demo, orchestrating sinusoidal leg motion and camera tracking.
- `ik.py` encapsulates planar and 3DOF inverse-kinematics routines reused by the viewer and command-line probes.
- `model/world.xml` references `model/robot.xml`, which in turn links meshes from `model/assets/`; modify assets or XMLs together to keep joint names stable.
- `model/openscad/` stores editable CAD sources alongside generated `.stl` files; regenerate meshes here before copying into `model/assets/`.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` creates an isolated environment for MuJoCo development.
- `pip install mujoco numpy` installs the only required Python dependencies seen in the scripts.
- `python height_control.py` launches the passive MuJoCo viewer with the oscillating height demo.
- `python ik.py` runs analytical IK spot-checks that cover both planar and tilted-leg modes.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, line length ≤ 100, and module-level constants in upper snake case.
- Functions and variables stay in snake_case (`solve_leg_ik_3dof`); keep short, descriptive docstrings explaining frames and units.
- Group imports into standard library, third-party, and local blocks, mirroring the existing files.
- XML and SCAD filenames use lowercase words separated by underscores; keep joint and body names stable to avoid MuJoCo warnings.

## Testing Guidelines
- Extend the lightweight checks in `ik.py` into `pytest` cases under `tests/` (name files `test_*.py`) to guard new IK modes.
- Before publishing changes that touch simulation parameters, run `python height_control.py` and observe joint limits, console warnings, and viewer stability.
- For headless verification, run `MUJOCO_GL=egl python height_control.py` to ensure the demo survives CI-like environments.

## Commit & Pull Request Guidelines
- Match the repository history by writing brief, present-tense commit subjects (e.g., `adding primitives model`); scope to a single concern per commit.
- Reference issues or design documents in the body when available, and note MuJoCo version or asset regeneration steps when relevant.
- Pull requests should summarise scenario coverage, list modified XML/STL paths, and attach screenshots or short clips when visual behaviour changes.

## Simulation Assets & Configuration Tips
- When tweaking kinematics, update link lengths in code, XML, and any derived constants together to avoid drift between simulation and analysis.
- Re-export SCAD models with `openscad -o model/assets/<part>.stl model/openscad/<part>.scad` and verify mesh orientation before committing.
