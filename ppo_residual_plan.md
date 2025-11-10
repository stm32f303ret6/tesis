# PPO Residual Training Plan

## Objectives
- Extend the existing Bézier gait (`gait_controller.py`) and IK stack (`ik.py`) with a residual action space so RL only corrects the nominal swing/stance targets.
- Build a reproducible Gymnasium environment that loads `model/world_train.xml`, exposes the MuJoCo-backed sensors already defined in `model/robot.xml`, and keeps controller timing identical to `height_control.py`.
- Train and evaluate a Stable-Baselines3 PPO policy that outputs per-leg residuals, logs diagnostics, and exports checkpoints ready for playback in the MuJoCo demo.

## Assumptions & Inputs
- Baseline scripts (`height_control.py`, `gait_controller.py`) already produce stable flat-ground motion in `model/world.xml`.
- `model/world_train.xml` introduces rough terrain but maintains identical actuator naming and control limits.
- Robot sensors declared in `model/robot.xml` (IMU, joint pos/vel, foot contacts, etc.) are accessible through MuJoCo's `data.sensordata`.
- Training occurs locally with the provided virtualenv + `requirements.txt`; SB3 (`stable-baselines3`), `gymnasium[mujoco]`, and `tensorboard` will be added as new dependencies.

## Phase 1 – Controller & Simulation Preparation
1. **Modularize controller hooks**
   - Extract the core control loop from `height_control.py` into a reusable class (e.g., `controllers/bezier_controller.py`) that exposes methods for setting body velocity targets and retrieving nominal foot positions per leg.
   - Ensure the controller produces foot targets in a consistent coordinate frame (likely world or hip frame) so the residual policy can operate in Cartesian deltas.
2. **IK interface audit**
   - Verify `ik.solve_leg_ik_3dof` handles perturbed targets without violating joint limits; add saturation logic if residuals can push feet outside reachable workspace.
   - Add lightweight unit tests (under `tests/`) that feed extreme residuals to IK to catch NaNs before RL training.
3. **World configuration**
   - Confirm both XML worlds share the same actuator order; document any extra bodies/geoms in `model/world_train.xml`.
   - Define a config dataclass (e.g., `SimulationConfig`) capturing MuJoCo paths, control timestep, and terrain toggles to keep scripts in sync.

## Phase 2 – Gymnasium Environment Design
1. **Environment skeleton**
   - Create `envs/residual_walk_env.py` with a `ResidualWalkEnv(gym.Env)` class that loads MuJoCo via `mujoco_py` or `mujoco` bindings used elsewhere.
   - Mirror the sim loop from `height_control.py`: call the Bézier controller, add residuals, run IK, set `data.ctrl`, and advance `sim.step()`.
2. **Observation space**
   - Stack base orientation/linear velocity, joint positions/velocities, phase timers, foot contact bits, and optionally terrain height samples.
   - Normalize each channel (e.g., clamp IMU quaternions, scale joint velocities) and capture limits in `gym.spaces.Box`.
3. **Action space**
   - Define a per-foot 3D residual (x/y/z) leading to a `Box(low=-residual_max, high=residual_max, shape=(12,))`.
   - Apply residuals in the controller before IK: `target = bezier_target + residual`, optionally filtered by a low-pass or blending factor.
4. **Reward shaping**
   - Core terms: forward velocity tracking, energy penalty (sum of |ctrl|), foot slip penalty (lateral velocity during stance), body attitude deviation, and termination bonus for staying upright.
   - Include sparse penalties for collisions or torso roll/pitch beyond limits; terminate episodes if the base height drops below a threshold.
5. **Episode & reset handling**
   - Implement deterministic resets by rewinding MuJoCo state and resetting controller phase timers.
   - Allow randomized terrain seeds or base perturbations to improve robustness (optional parameter in `reset()`).

## Phase 3 – Training Pipeline
1. **Dependencies & configuration**
   - Add SB3, Gymnasium, TensorBoard, and `mujoco` Python binding requirements; document install command in `README` or `AGENTS.md`.
   - Create a `configs/ppo_residual.yaml` capturing hyperparameters (learning rate, batch size, clip range, residual scale, reward weights).
2. **Training script**
   - Implement `train_residual_ppo.py` that:
     - Creates the environment with vectorized wrappers (`SubprocVecEnv` or `AsyncVectorEnv`) and optional `VecNormalize`.
     - Instantiates `PPO` with an MLP policy; configure observation normalization, entropy bonus, and callback list.
     - Saves checkpoints, normalizer stats, and TensorBoard logs under `runs/residual_ppo/YYYYMMDD_HHMM`.
3. **Callbacks & logging**
   - Add custom callbacks for curriculum adjustment (increase terrain difficulty), evaluation rollouts in `model/world.xml`, and early stopping on plateaued rewards.
   - Log reward components, falls, and controller saturation metrics for debugging.

## Phase 4 – Evaluation & Integration
1. **Playback script**
   - Extend `height_control.py` or add `play_residual_policy.py` to load a trained PPO checkpoint, run inference (batched or single-step), and visualize in MuJoCo.
   - Provide CLI flags for selecting `world.xml` vs `world_train.xml`, toggling residual usage, and adjusting action scaling.
2. **Automated regression hooks**
   - Update `tests/compare_world_trajectories.py` to optionally run with the residual policy and dump PNG comparisons for review.
   - Add a smoke test (`tests/test_residual_env.py`) that instantiates the Gym env, runs a random policy for a few steps, and ensures no asserts.
3. **Documentation**
   - Document training commands, hyperparameters, and expected GPU/CPU requirements.
   - Capture reward definitions and termination conditions in the README or a new `docs/residual_learning.md`.

## Phase 5 – Validation & Next Steps
- **Validation runs**: Evaluate trained policies on both flat and rough worlds, logging success rate and average distance.
- **Ablation studies**: Compare residual vs. direct torque control to highlight stability benefits.
- **Safety checks**: Add assertions for joint limit violations and actuator saturation before policy deployment on hardware.
- **Future extensions**: Curriculum over terrain difficulty, domain randomization (mass, friction), and transferring residual policy to real robot via system ID.
