# Repository Guidelines

## Project Structure & Module Organization
- Root scripts drive day-to-day work: `height_control.py` launches the MuJoCo demo, `gait_controller.py` carries the diagonal gait state machine, and `ik.py` owns the SCARA solvers shared by sim and tooling.
- Utilities such as `foot_range_calculator.py` (workspace sweeps) and the `tests/compare_world_trajectories.py` harness surface regressions; consult `CLAUDE.md` for safe parameter envelopes.
- All scenes, robot definitions, and meshes live under `model/`; keep body/joint names stable because controllers index MuJoCo objects by string.
- The `tests/` folder stores regression scripts, its README, and the tracked `trajectory_comparison.png`.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` – boots the MuJoCo + NumPy toolchain.
- `python3 height_control.py` – interactive trot demo (`MUJOCO_GL=egl` for headless nodes or CI).
- `python3 foot_range_calculator.py` – verifies reachable volumes after geometry or parameter edits.
- `python3 tests/compare_world_trajectories.py` – reruns both worlds and refreshes `tests/trajectory_comparison.png`.
- `python3 ik.py` – smoke-test for solver math; run before pushing IK changes.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indents, ≤100-character lines, snake_case identifiers, and UPPER_SNAKE_CASE constants (`LEG_NAMES`, `IK_PARAMS`). Dataclasses (e.g., `GaitParameters`) group tunables.
- Use type hints, short docstrings that state frames/units, and grouped imports (stdlib → third-party → local). Prefer vectorized NumPy operations and side-effect-light helpers to ease testing.

## Testing Guidelines
- Re-run `python3 ik.py` and `python3 foot_range_calculator.py` whenever IK math, link lengths, or gait limits move; attach logs in review when behavior shifts.
- Execute `python3 tests/compare_world_trajectories.py` from the repo root. Treat unexpected PNG diffs as red flags and note intentional ones in PR descriptions.
- Add new pytest-style modules in `tests/` (pattern `test_*.py`) with deterministic seeding for any stochastic sampling.

## Commit & Pull Request Guidelines
- Match the existing short, present-tense subjects (`adding rough terrain`, `adding foot range calculator`) and elaborate in bodies when touching MuJoCo XML, controller timing, or dependencies.
- Each PR should state the behavior change, list reproduction commands, include updated plots/screenshots, link relevant issues, and call out required env vars (e.g., `MUJOCO_GL=egl`, MuJoCo keys).

## Simulation Assets & Configuration Tips
- Regenerate meshes through `model/openscad/` and synchronize any dimension shift with `IK_PARAMS`/`GAIT_PARAMS` plus the XML files to avoid drift.
- Keep MuJoCo licenses in `~/.mujoco`; for headless automation export `MUJOCO_GL=egl` or `osmesa` before running the simulator to prevent GL backend failures.
