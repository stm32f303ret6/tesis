#!/usr/bin/env python3
"""Phase 2 viewer-based evaluation: walk and log obs/rewards for 20s.

This script mirrors height_control.py's viewer-driven loop but augments it with
observation and reward logging aligned with Phase 2 specs. It uses the
BezierGaitResidualController with zero residuals to reproduce the baseline gait
on flat ground and evaluates reward components while rendering.

Outputs (written to phases_test/):
- phase2_env_spec.txt              – observation dim, duration, timestep
- phase2_step_trace.csv            – time, reward, fwd_vel, height, terminated, truncated
- phase2_reward_components.json    – mean reward component values

Usage
- Headless: export MUJOCO_GL=egl
- Run:      python3 phases_test/phase2_viewer_walk_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
import sys
from typing import Dict, Tuple

import mujoco
import mujoco.viewer
import numpy as np

# Ensure repo root is on path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.bezier_gait_residual import BezierGaitResidualController
from gait_controller import GaitParameters, LEG_NAMES
from ik import solve_leg_ik_3dof
from utils.control_utils import apply_leg_angles
from utils.sensor_utils import SensorReader


# Keep in sync with height_control.py and envs/residual_walk_env.py
IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0  # +1 keeps controller +X, -1 flips to match leg IK frame

GAIT_PARAMS = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = [float(q) for q in quat]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=float,
    )
    return R


def quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """Return roll, pitch, yaw from MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = [float(q) for q in quat]
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def build_foot_contact_bodies(model: mujoco.MjModel) -> Dict[str, Tuple[int, int]]:
    """Return mapping leg->(elbow_left_bodyid, elbow_right_bodyid)."""
    mapping: Dict[str, Tuple[int, int]] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        bl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_left")
        br = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_right")
        mapping[leg] = (int(bl) if bl != -1 else -1, int(br) if br != -1 else -1)
    return mapping


def get_foot_contact_flags(model: mujoco.MjModel, data: mujoco.MjData, foot_bodies: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """Heuristic contact indicator per leg (0.0 or 1.0)."""
    forces = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
    if data.ncon <= 0:
        return forces

    geom_body = model.geom_bodyid
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1 = int(geom_body[g1]) if g1 >= 0 else -1
        b2 = int(geom_body[g2]) if g2 >= 0 else -1
        for leg, (bl, br) in foot_bodies.items():
            if bl == -1 and br == -1:
                continue
            if b1 in (bl, br) or b2 in (bl, br):
                forces[leg] = 1.0
    return forces


def compute_observation(sensor: SensorReader, controller: BezierGaitResidualController) -> np.ndarray:
    """Compose observation vector per Phase 2 design."""
    obs_components = []

    body_state = sensor.get_body_state()
    _, quat, linvel, angvel = np.split(body_state, [3, 7, 10])
    quat = quat / max(1e-9, float(np.linalg.norm(quat)))
    obs_components.append(quat)
    obs_components.append(linvel)
    obs_components.append(angvel)

    gravity_world = np.array([0.0, 0.0, -1.0])
    R = quat_to_rotation_matrix(quat)
    gravity_body = R.T @ gravity_world
    obs_components.append(gravity_body)

    obs_components.append(sensor.get_joint_states())
    obs_components.append(sensor.get_foot_positions())
    obs_components.append(sensor.get_foot_velocities())

    phase_info = controller.get_phase_info()
    phase = float(phase_info.get("phase_normalized", 0.0))
    obs_components.append(np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)], dtype=float))

    flags = controller.get_swing_stance_flags()
    flag_array = np.array([flags[leg] for leg in LEG_NAMES], dtype=float)
    obs_components.append(flag_array)

    # Previous action placeholder is included only for shape parity in env;
    # here we omit it, since this viewer script does not execute RL actions.
    # If needed, append zeros(12) to match env's exact dimension.
    obs_components.append(np.zeros(12, dtype=float))

    # Command velocity (match env default 0.2 m/s)
    obs_components.append(np.array([0.2], dtype=float))

    obs = np.concatenate(obs_components).astype(np.float32)
    if not np.all(np.isfinite(obs)):
        raise FloatingPointError("Non-finite values in observation")
    return obs


def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor: SensorReader,
    controller: BezierGaitResidualController,
    prev_action: np.ndarray,
    last_last_action: np.ndarray | None,
    foot_bodies: Dict[str, Tuple[int, int]],
    target_velocity: float = 0.2,
    target_height: float = 0.07,
) -> Tuple[float, Dict[str, float]]:
    """Compute reward with component breakdown (mirrors envs/residual_walk_env)."""
    rewards: Dict[str, float] = {}

    linvel = sensor.read_sensor("body_linvel")
    forward_vel = float(linvel[0])
    rewards["forward_velocity"] = 1.0 - abs(forward_vel - float(target_velocity))

    rewards["lateral_stability"] = -0.5 * abs(float(linvel[1]))

    body_pos = sensor.read_sensor("body_pos")
    rewards["height"] = -1.0 * abs(float(body_pos[2]) - float(target_height))

    quat = sensor.get_body_quaternion()
    roll, pitch, _ = quat_to_euler(quat)
    rewards["orientation"] = -1.0 * float(roll * roll + pitch * pitch)

    action_magnitude = float(np.linalg.norm(prev_action))
    rewards["energy"] = -0.1 * action_magnitude

    if last_last_action is not None:
        action_change = float(np.linalg.norm(prev_action - last_last_action))
        rewards["smoothness"] = -0.2 * action_change
    else:
        rewards["smoothness"] = 0.0

    foot_contacts = get_foot_contact_flags(model, data, foot_bodies)
    swing_flags = controller.get_swing_stance_flags()
    contact_reward = 0.0
    for leg in LEG_NAMES:
        is_swing = int(swing_flags[leg]) == 1
        has_contact = foot_contacts[leg] > 0.5
        if is_swing and has_contact:
            contact_reward -= 0.5
        elif is_swing and not has_contact:
            contact_reward += 0.1
        elif (not is_swing) and has_contact:
            contact_reward += 0.1
        else:
            contact_reward -= 0.5
    rewards["contact_pattern"] = float(contact_reward)

    joint_positions = sensor.get_joint_states()[:12]
    limit_violations = float(np.sum(np.abs(joint_positions) > 2.5))
    rewards["joint_limits"] = -1.0 * limit_violations

    total = float(sum(rewards.values()))
    return total, rewards


def apply_gait_targets(model: mujoco.MjModel, data: mujoco.MjData, controller: BezierGaitResidualController, timestep: float) -> None:
    """Evaluate the gait planner and push the resulting joint targets to MuJoCo."""
    zero_residuals = {leg: np.zeros(3) for leg in LEG_NAMES}
    leg_targets = controller.update_with_residuals(timestep, zero_residuals)

    for leg in LEG_NAMES:
        target = np.asarray(leg_targets.get(leg), dtype=float)
        target_local = target.copy()
        target_local[0] *= FORWARD_SIGN

        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is None:
            # Keep previous command if IK fails
            continue
        apply_leg_angles(data, leg, result)


def run_sim(duration: float,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            sensor: SensorReader,
            controller: BezierGaitResidualController,
            robot_body_id: int,
            foot_bodies: Dict[str, Tuple[int, int]],
            csv_writer,
            target_velocity: float,
            target_height: float,
            use_viewer: bool = True) -> Tuple[int, float, Dict[str, float]]:
    """Core simulation loop used by both viewer and headless modes."""
    total_reward = 0.0
    n_steps = 0
    comp_sums: Dict[str, float] = {}
    prev_action = np.zeros(12, dtype=float)
    last_last_action: np.ndarray | None = None

    def log_step():
        nonlocal total_reward, n_steps, prev_action, last_last_action, comp_sums
        reward, comps = compute_reward(
            model, data, sensor, controller, prev_action, last_last_action, foot_bodies, target_velocity, target_height
        )
        total_reward += reward
        n_steps += 1
        for k, v in comps.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v)

        linvel = sensor.read_sensor("body_linvel")
        body_pos = sensor.read_sensor("body_pos")
        quat = sensor.get_body_quaternion()
        roll, pitch, _ = quat_to_euler(quat)
        terminated = bool((float(body_pos[2]) < 0.03) or (abs(roll) > math.pi / 3) or (abs(pitch) > math.pi / 3))
        truncated = bool(data.time >= duration)
        csv_writer.writerow([
            f"{data.time:.6f}",
            f"{reward:.6f}",
            f"{float(linvel[0]):.6f}",
            f"{float(body_pos[2]):.6f}",
            int(terminated),
            int(truncated),
        ])

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                step_start = time.time()
                apply_gait_targets(model, data, controller, model.opt.timestep)
                mujoco.mj_step(model, data)
                # Camera follow
                robot_pos = data.xpos[robot_body_id]
                viewer.cam.lookat[:] = robot_pos
                viewer.sync()
                log_step()
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        # Headless loop
        while data.time < duration:
            step_start = time.time()
            apply_gait_targets(model, data, controller, model.opt.timestep)
            mujoco.mj_step(model, data)
            log_step()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    return n_steps, total_reward, comp_sums


def main() -> None:
    duration = 20.0  # seconds
    target_velocity = 0.2
    target_height = 0.07

    model = mujoco.MjModel.from_xml_path("model/world_train.xml")
    data = mujoco.MjData(model)
    sensor = SensorReader(model, data)

    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    controller = BezierGaitResidualController(params=GAIT_PARAMS, residual_scale=0.02)
    controller.reset()

    out_dir = Path("phases_test")
    out_dir.mkdir(exist_ok=True)
    spec_path = out_dir / "phase2_env_spec.txt"
    csv_path = out_dir / "phase2_step_trace.csv"
    comps_path = out_dir / "phase2_reward_components.json"

    # Precompute for contacts
    foot_bodies = build_foot_contact_bodies(model)

    # Warm-up obs to record dimension
    obs = compute_observation(sensor, controller)
    with spec_path.open("w") as f:
        f.write(f"observation_dim: {obs.size}\n")
        f.write(f"action_dim (residuals): 12\n")
        f.write(f"duration_s: {duration}\n")
        f.write(f"timestep_s: {model.opt.timestep}\n")

    # CSV setup
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["time", "reward", "forward_velocity", "body_height", "terminated", "truncated"])  # header
    # Decide rendering mode: prefer headless for EGL/OSMESA or when NO_RENDER is set.
    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    have_display = bool(os.environ.get("DISPLAY"))
    allow_render = os.environ.get("NO_RENDER", "0") not in ("1", "true", "True")
    # Use viewer only if using GLFW backend and a DISPLAY is available and rendering allowed
    use_viewer_env = allow_render and have_display and (gl_backend in ("", "glfw"))
    try:
        n_steps, total_reward, comp_sums = run_sim(
            duration,
            model,
            data,
            sensor,
            controller,
            robot_body_id,
            foot_bodies,
            writer,
            target_velocity,
            target_height,
            use_viewer=use_viewer_env,
        )
    except Exception as e:
        # Likely a GLFW/display init error – retry headless
        print(f"Viewer unavailable ({e}); falling back to headless stepping...")
        n_steps, total_reward, comp_sums = run_sim(
            duration,
            model,
            data,
            sensor,
            controller,
            robot_body_id,
            foot_bodies,
            writer,
            target_velocity,
            target_height,
            use_viewer=False,
        )

    csv_file.close()

    # Write mean reward components
    comp_means = {k: (v / n_steps) for k, v in comp_sums.items()} if n_steps > 0 else {}
    with comps_path.open("w") as f:
        json.dump(comp_means, f, indent=2)

    print("\nPhase 2 viewer evaluation summary:")
    print(f"  Steps: {n_steps}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Mean reward/step: { (total_reward / max(1, n_steps)):.4f}")
    print(f"  Spec: {spec_path}")
    print(f"  Trace: {csv_path}")
    print(f"  Reward components: {comps_path}")


if __name__ == "__main__":
    main()
