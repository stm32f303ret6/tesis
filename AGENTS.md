# Repository Guidelines

This repository contains Python code and MuJoCo models for a parallel SCARA leg demo and IK utilities. Use this guide to develop, test, and submit changes consistently.

## Project Structure & Module Organization
- Top-level scripts: `ik.py` (IK solvers), `height_control.py` (sinusoidal height demo), `smooth_startup.py` (eased startup + height control).
- Models and assets: `model/world.xml`, `model/robot.xml`, `model/assets/*.stl`, `model/openscad/*.scad`.
- Run scripts from the repo root so relative paths (e.g., `model/world.xml`) resolve correctly.

## Build, Test, and Development Commands
- Environment setup (recommended):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -U mujoco numpy`
- Run demos:
  - `python3 height_control.py` (live viewer)
  - `python3 smooth_startup.py` (smooth tilt → height control)
- Optional tools:
  - `pip install black ruff mypy`
  - Format: `black .`  Lint: `ruff .`  Type-check: `mypy ik.py`

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indentation.
- Names: modules/files `snake_case`, functions `snake_case`, constants `UPPER_CASE`.
- Prefer type hints for public functions in `ik.py`; keep functions small and pure (no I/O).
- Keep viewer/demo logic in scripts; keep math/IK logic in `ik.py`.

## Testing Guidelines
- Framework: pytest. Add tests under `tests/` with `test_*.py` naming.
- Focus on numeric IK assertions (reachable vs. unreachable, angle ranges).
- Example: `python -m pytest -q` (aim for >80% coverage of core IK logic).

## Commit & Pull Request Guidelines
- Use Conventional Commits style: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs should include: concise description, steps to run (`python3 smooth_startup.py`), any visuals if helpful, and notes on model/asset changes.
- Keep changes focused; avoid breaking relative paths to `model/*`.

## Security & Configuration Tips
- MuJoCo viewer requires a working OpenGL context. Headless servers: `export MUJOCO_GL=egl` before running.
- Do not commit large binaries beyond `model/assets/*.stl`; keep SCAD/STL paths stable.
- When changing IK defaults, update usage examples in `ik.py` and verify demos still run.
