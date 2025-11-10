# PPO Residual Learning – Phase-by-Phase Tracker

This document tracks implementation and verification of the PPO residual learning plan (see `ppo_residual_plan_detailed.md`). It is organized into phases with:
- What to implement (files to add/change)
- How to test (commands to run)
- Expected outputs (artifacts under `phases_test/`)
- Acceptance criteria (checklist)

Notes
- No pyenv/venv: use system Python and already-installed libraries.
- MuJoCo headless nodes: export `MUJOCO_GL=egl` before running sims/train.
- All verification artifacts are written inside this folder for easy diffing.

Quick Links
- Sensors available in `model/robot.xml` (body, joints, feet) back this plan.
- World files: `model/world.xml` (flat), `model/world_train.xml` (terrain).

---

## Phase 1 — Controller Wrapper + Sensor Utilities

Goal: Wrap Bézier gait with residual support and provide a clean sensor API.

Implement
- `controllers/bezier_gait_residual.py`
  - `BezierGaitResidualController` with:
    - `residual_scale` (default 0.02 m)
    - `update_with_residuals(dt, residuals)` → adds clipped residuals to base targets
    - `get_phase_info()` and `get_swing_stance_flags()`
- `utils/sensor_utils.py`
  - `SensorReader(model, data)` able to read:
    - `body_pos`, `body_quat`, `body_linvel`, `body_angvel`
    - Joint pos/vel for all actuated joints: `*_tilt_pos|vel`, `*_shoulder_L_pos|vel`, `*_shoulder_R_pos|vel`
    - Foot pos/vel: `{FL,FR,RL,RR}_foot_pos|vel`
- `utils/control_utils.py`
  - Move `apply_leg_angles(...)` from `height_control.py` for reuse (keep call sites working).

Test
- Command: `python3 phases_test/phase1_verify_controller.py`
  - Verifies residual clipping (max |residual| ≤ 0.02), zero residual preserves base,
    and all expected sensors exist.

Expected outputs
- `phases_test/phase1_controller_summary.json`
  - `{ "zero_residuals_preserve_base": true, "max_abs_residual_applied": <=0.02, "sensors_ok": true, "observed_sensors": [...] }`

Acceptance
- [ ] Residual wrapper returns per-leg 3D targets with clipping
- [ ] SensorReader confirms all names in `model/robot.xml` are readable
- [ ] Summary JSON generated with all flags true

---

## Phase 2 — Gymnasium Environment

Goal: A custom Gymnasium env that consumes sensors, adds residuals to gait targets, runs IK, and steps MuJoCo.

Implement
- `envs/residual_walk_env.py` with:
  - Observation (~70–80D): quat(4), lin vel(3), ang vel(3), projected gravity(3), joint pos/vel(24), phase sin/cos(2), swing/stance(4), prev action(12), command vel(1)
  - Action (12D): per-foot residuals in [-1,1], scaled by `residual_scale`
  - Reward: forward velocity tracking + stability/height/orientation/energy/smoothness/contact pattern/joint limit penalty
  - Termination: low height or excessive roll/pitch; truncation by time limit
  - Reset: optional small state randomization, controller reset, short settle
  - Step: process action → controller residuals → IK (`solve_leg_ik_3dof`) → `apply_leg_angles` → `mujoco.mj_step`

Test
- Command: `MUJOCO_GL=egl python3 phases_test/phase2_env_smoke.py`
  - Runs 100 steps with zero actions, then 100 with random actions; logs spec and rewards.

Expected outputs
- `phases_test/phase2_env_spec.txt` (dims of obs/action)
- `phases_test/phase2_step_trace.csv` (step,total_reward,forward_velocity,body_height,terminated,truncated)
- `phases_test/phase2_reward_components.json` (mean of each reward term)

Acceptance
- [ ] Observation shape matches spec; action shape is (12,)
- [ ] No NaN/Inf in observations or rewards across 200 steps
- [ ] Trace and JSON files present with plausible values

---

## Phase 3 — PPO Training Pipeline (Smoke)

Goal: Vectorized PPO training with normalization; verify end-to-end on a short run.

Implement
- `train_residual_ppo.py` (SB3 PPO + `DummyVecEnv` + `VecNormalize`, tensorboard logging)
- Optional: `callbacks/curriculum_callback.py` with `set_terrain_scale(scale)` stub in env

Test
- Command: `bash phases_test/phase3_training_smoke.sh`
  - Runs `total_timesteps=10000`, `n_envs=1`, `n_steps=512`, writes to `runs/smoke_*`.

Expected outputs
- `phases_test/phase3_artifacts.txt` listing:
  - `runs/smoke_*/final_model.zip`
  - `runs/smoke_*/vec_normalize.pkl`
- `phases_test/phase3_training_smoke_metrics.json` with: `{ "timesteps": 10000, "tensorboard_logdir_exists": true, "episode_reward_mean_present": true }`

Acceptance
- [ ] Model and normalization artifacts created
- [ ] Training completes without exceptions
- [ ] TB scalars present (episode reward/len)

---

## Phase 4 — Evaluation & Visualization

Goal: Load policy for playback and compare against baseline Bézier-only.

Implement
- `play_residual_policy.py` (loads PPO model, optional `VecNormalize`, runs episodes)
- `tests/compare_residual_vs_baseline.py` (bar charts: distance, fall rate)

Test
- Command: `bash phases_test/phase4_playback.sh`
  - Uses smoke model from Phase 3; runs couple of episodes headless, then comparison.

Expected outputs
- `phases_test/phase4_playback.txt` (episode reward/steps/height/velocity per episode)
- `phases_test/residual_vs_baseline.png` (comparison plot)

Acceptance
- [ ] Playback script runs without rendering
- [ ] Comparison image generated
- [ ] Residual mode executes actions; baseline uses zeros

---

## Phase 5 — Hyperparameter Tuning (Optional)

Goal: Short Optuna sweep to validate tuning loop and record best params.

Implement
- `tune_hyperparameters.py` (Optuna + short training per trial)

Test
- Command: `bash phases_test/phase5_tune_smoke.sh` (e.g., 5 trials × 30k steps)

Expected outputs
- `phases_test/phase5_tune_topk.csv` (trial, lr, ent_coef, gamma, mean_reward)
- `phases_test/phase5_best_params.json` (best hyperparameters)

Acceptance
- [ ] Tuning completes with top-k CSV and best params JSON

---

## Phase 6 — Tests & Validation

Goal: Unit/integration tests for env correctness and SB3 training smoke behavior.

Implement
- `tests/test_residual_env.py` (create/reset/step/zero-residual stability/clipping)
- `tests/test_training_smoke.py` (short learn; save/load; predict parity)

Test
- Command: `pytest -q` (or `pytest tests/test_residual_env.py -v`)

Expected outputs
- `phases_test/phase6_pytest_results.txt` (captured log with pass summary)
- `phases_test/phase6_failures.txt` (only if failures occur)

Acceptance
- [ ] Env tests pass
- [ ] Training smoke tests pass (if RL libs installed)

---

## Phase 7 — Deployment Checklist & Monitoring

Goal: Validate success criteria on rough terrain and export training curves.

Implement
- `utils/monitor_training.py` (reads TB logs → `training_curves.png`)
- `phases_test/phase7_validate.py` (runs N eval episodes, computes metrics)

Test
- Command: `MUJOCO_GL=egl python3 phases_test/phase7_validate.py --model runs/<run>/best_model/best_model.zip --normalize runs/<run>/vec_normalize.pkl`

Expected outputs
- `phases_test/training_curves.png`
- `phases_test/phase7_validation.json` with keys:
  - `success_rate` (target ≥ 0.8), `avg_episode_length` (≥ 500), `height_within_2cm`, `vel_within_0.1ms`, `no_nans`

Acceptance
- [ ] Metrics satisfy thresholds
- [ ] Curves exported

---

## Appendix — Sensor Names (from model/robot.xml)

Body
- `body_pos`, `body_quat`, `body_linvel`, `body_angvel`

Joint positions
- `FL_tilt_pos`, `FL_shoulder_L_pos`, `FL_shoulder_R_pos`
- `FR_tilt_pos`, `FR_shoulder_L_pos`, `FR_shoulder_R_pos`
- `RL_tilt_pos`, `RL_shoulder_L_pos`, `RL_shoulder_R_pos`
- `RR_tilt_pos`, `RR_shoulder_L_pos`, `RR_shoulder_R_pos`

Joint velocities
- `FL_tilt_vel`, `FL_shoulder_L_vel`, `FL_shoulder_R_vel`
- `FR_tilt_vel`, `FR_shoulder_L_vel`, `FR_shoulder_R_vel`
- `RL_tilt_vel`, `RL_shoulder_L_vel`, `RL_shoulder_R_vel`
- `RR_tilt_vel`, `RR_shoulder_L_vel`, `RR_shoulder_R_vel`

Feet
- Positions: `FL_foot_pos`, `FR_foot_pos`, `RL_foot_pos`, `RR_foot_pos`
- Velocities: `FL_foot_vel`, `FR_foot_vel`, `RL_foot_vel`, `RR_foot_vel`

---

## Running Notes

- Use `python3` everywhere (no virtual env). If some deps are missing, install them system-wide as needed.
- Headless rendering: `export MUJOCO_GL=egl` before running env/training.
- Keep `IK_PARAMS` and `FORWARD_SIGN` consistent with `height_control.py` when wiring the env.
- All verification scripts referenced above will write into this folder; you can commit artifacts to track regressions.

