# Repository Guidelines

## Project Structure & Module Organization
- Root scripts drive daily workflows:
  - `height_control.py` — launches the MuJoCo trot demo.
  - `gait_controller.py` — diagonal gait state machine.
  - `ik.py` — SCARA IK solvers shared by sim and tooling.
- Utilities: `foot_range_calculator.py` (workspace sweeps) and `tests/compare_world_trajectories.py` (regression harness). See `CLAUDE.md` for safe parameter envelopes.
- Assets: all scenes, robot XML, and meshes live under `model/` (including `model/openscad/`). Keep body/joint names stable; controllers index MuJoCo objects by string.
- Code modules: `controllers/`, `envs/`, `utils/`, and `callbacks/` hold reusable pieces. Tests and artifacts live in `tests/` (tracked `tests/trajectory_comparison.png`).

## Build, Test, and Development Commands
- Bootstrap: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Demo (headless via EGL): `MUJOCO_GL=egl python3 height_control.py`
- IK smoke test: `python3 ik.py`
- Foot workspace check: `python3 foot_range_calculator.py`
- Regression plot refresh: `python3 tests/compare_world_trajectories.py` (run from repo root)
- MuJoCo keys belong in `~/.mujoco`. For CI/headless, export `MUJOCO_GL=egl` (or `osmesa`).

## Coding Style & Naming Conventions
- PEP 8, 4‑space indents, ≤100‑char lines. Snake_case for vars/functions; UPPER_SNAKE_CASE for constants (e.g., `LEG_NAMES`, `IK_PARAMS`).
- Use type hints and short docstrings that state frames/units. Group imports: stdlib → third‑party → local.
- Prefer vectorized NumPy and side‑effect‑light helpers. Use dataclasses (e.g., `GaitParameters`) to group tunables.

## Testing Guidelines
- Re‑run `python3 ik.py` and `python3 foot_range_calculator.py` when IK math, link lengths, or gait limits change; attach logs when behavior shifts.
- Run `python3 tests/compare_world_trajectories.py`; treat unexpected PNG diffs as red flags and document intentional changes in PRs.
- Add pytest‑style tests in `tests/` named `test_*.py`; seed randomness deterministically.

## Commit & Pull Request Guidelines
- Commits: short, present‑tense subjects (e.g., “adding foot range calculator”); expand in bodies for MuJoCo XML, controller timing, or dependency changes.
- PRs include: behavior change summary, reproduction commands, updated plots/screenshots, linked issues, and required env vars (e.g., `MUJOCO_GL=egl`, MuJoCo keys).

## Simulation Assets & Configuration Tips
- Regenerate meshes in `model/openscad/`; keep XML, `IK_PARAMS`, and `GAIT_PARAMS` in sync to avoid drift.
- Maintain stable body/joint names across XML and controllers.
