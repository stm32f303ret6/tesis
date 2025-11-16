#!/usr/bin/env python3
"""Visualize and evaluate trained adaptive gait policy.

This script loads a trained adaptive gait policy and runs it in the MuJoCo
viewer, displaying real-time gait parameter adaptations.

Usage:
    # Run with trained policy
    python3 play_adaptive_policy.py \\
        --model runs/adaptive_gait_XXX/final_model.zip \\
        --normalize runs/adaptive_gait_XXX/vec_normalize.pkl \\
        --seconds 30 \\
        --deterministic

    # Run baseline (pure gait controller without residuals)
    python3 play_adaptive_policy.py --baseline --seconds 30
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import os
import sys

import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.adaptive_gait_env import AdaptiveGaitEnv
from gait_controller import GaitParameters

RESIDUAL_SCALE = 0.01
def make_env():
    """Create evaluation environment."""
    gait = GaitParameters(
        body_height=0.05,
        step_length=0.06,
        step_height=0.04,
        cycle_time=0.8
    )
    return AdaptiveGaitEnv(
        model_path="model/world_train.xml",
        gait_params=gait,
        residual_scale=RESIDUAL_SCALE,
        max_episode_steps=60000,
        settle_steps=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Play trained adaptive gait policy")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.zip)")
    parser.add_argument("--normalize", type=str, default=None, help="Path to VecNormalize stats (.pkl)")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duration to run [seconds]")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--flat", action="store_true", help="Use flat terrain instead of rough")
    parser.add_argument("--no-reset", action="store_true", help="Disable automatic reset on termination (for visualization)")
    parser.add_argument("--baseline", action="store_true", help="Run baseline gait without residuals (zero actions)")
    parser.add_argument("--save-trajectory", type=str, default=None, help="Save trajectory data to JSON file")
    parser.add_argument("--fullscreen", action="store_true", help="Start viewer in fullscreen (GLFW backend only)")
    args = parser.parse_args()

    # Fullscreen feasibility checks
    if args.fullscreen:
        backend = os.environ.get("MUJOCO_GL", "glfw").lower()
        if backend != "glfw":
            print(f"Warning: --fullscreen requires MUJOCO_GL=glfw (current: {backend}); attempting maximize fallback.")
        if sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"):
            print("Warning: DISPLAY is not set; viewer may not open. Fullscreen unavailable in headless mode.")

    # Load model (unless baseline mode)
    model = None
    if args.baseline:
        print("Running in BASELINE mode (zero actions - pure gait controller)")
    else:
        if args.model is None:
            raise SystemExit("--model is required unless --baseline is set")
        print(f"Loading model from {args.model}...")
        model = PPO.load(args.model)

    # Determine terrain model path
    if args.flat:
        model_path = "model/world.xml"
        terrain_name = "flat (world.xml)"
    else:
        model_path = "model/world_train.xml"
        terrain_name = "rough (world_train.xml)"

    # Create environment
    print(f"Creating environment with {terrain_name}...")
    env = AdaptiveGaitEnv(
        model_path=model_path,
        gait_params=GaitParameters(
            body_height=0.05,
            step_length=0.067,
            step_height=0.04,
            cycle_time=0.9
        ),
        residual_scale=RESIDUAL_SCALE,
        max_episode_steps=60000,
        settle_steps=0,
    )

    # Wrap for normalization if stats provided
    if args.normalize:
        print(f"Loading normalization stats from {args.normalize}...")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False  # Disable running mean updates
        vec_env.norm_reward = False  # Don't normalize rewards during eval
        use_vec_env = True
    else:
        print("No normalization stats provided, using raw observations")
        use_vec_env = False

    print("\n" + "=" * 80)
    print("Starting playback...")
    print("=" * 80)
    print(f"Duration:        {args.seconds:.1f}s")
    print(f"Deterministic:   {args.deterministic}")
    print(f"Terrain:         {'flat (world.xml)' if args.flat else 'rough (world_train.xml)'}")
    print(f"Auto-reset:      {'disabled' if args.no_reset else 'enabled'}")
    print(f"Fullscreen:      {'enabled' if args.fullscreen else 'disabled'}")
    print(f"Mode:            {'BASELINE (zero actions - pure gait)' if args.baseline else f'ADAPTIVE (residual_scale={RESIDUAL_SCALE})'}")
    print("\nWatch the console for real-time gait parameter updates!")
    print("Controls:        Press SPACE to pause/unpause simulation")
    print("=" * 80)
    print()

    # Pause state (shared between keyboard callback and main loop)
    paused = {"state": False}

    def key_callback(keycode):
        """Handle keyboard input for pause/unpause."""
        # Space key code is 32 in MuJoCo
        if keycode == 32:  # Space key
            paused["state"] = not paused["state"]
            status = "PAUSED" if paused["state"] else "RESUMED"
            print(f"\n[Simulation {status}]\n")

    # Open MuJoCo viewer (hide side UIs when requesting fullscreen for a cleaner look)
    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        show_left_ui=not args.fullscreen,
        show_right_ui=not args.fullscreen,
        key_callback=key_callback,
    ) as viewer:
        # Optional fullscreen handling (GLFW backend only)
        if args.fullscreen:
            try:
                # Use MuJoCo-provided glfw bindings to avoid extra dependency
                from mujoco import glfw  # type: ignore

                # Try to locate the GLFW window handle via known attributes
                window = None
                # 1) Direct attributes on Handle (unlikely on 3.3+)
                for name in ("_window", "window"):
                    window = getattr(viewer, name, None)
                    if window is not None:
                        break
                # 2) Reach into internal simulate object (private) and common names
                if window is None and hasattr(viewer, "_get_sim"):
                    sim = None
                    try:
                        sim = viewer._get_sim()  # type: ignore[attr-defined]
                    except Exception:
                        sim = None
                    for name in ("glfw_window", "_glfw_window", "window", "_window", "context", "gl_context"):
                        if sim is None:
                            break
                        cand = getattr(sim, name, None)
                        # unwrap context objects that may hold the window
                        if cand is not None and name in ("context", "gl_context"):
                            cand = getattr(cand, "window", None)
                        if cand is not None:
                            window = cand
                            break
                if window is None:
                    raise RuntimeError("Viewer window handle not available")

                monitor = glfw.get_primary_monitor()
                if not monitor:
                    raise RuntimeError("No primary monitor detected (headless or backend not GLFW)")

                mode = glfw.get_video_mode(monitor)
                if mode is None:
                    raise RuntimeError("Unable to query monitor video mode")

                # Support both PyGLFW structures: mode.size.width or mode.width
                if hasattr(mode, "size"):
                    width, height = int(mode.size.width), int(mode.size.height)
                else:
                    width, height = int(getattr(mode, "width")), int(getattr(mode, "height"))

                refresh = int(getattr(mode, "refresh_rate", getattr(glfw, "DONT_CARE", 0)))

                glfw.set_window_monitor(window, monitor, 0, 0, width, height, refresh)
                print(f"Fullscreen mode enabled ({width}x{height}@{refresh or 'NA'}Hz)")
            except Exception as e:
                # Fall back to maximize if fullscreen fails
                try:
                    from mujoco import glfw  # type: ignore
                    # Try maximizing using any found window handle
                    win = None
                    for obj in (viewer, getattr(viewer, "_get_sim", lambda: None)()):
                        for name in ("_window", "window", "glfw_window", "_glfw_window"):
                            win = getattr(obj, name, None) if obj is not None else None
                            if win is not None:
                                break
                        if win is not None:
                            break
                    if win is not None:
                        glfw.maximize_window(win)
                        print(f"Fullscreen not available ({e}); maximized window instead.")
                    else:
                        print(f"Fullscreen not available ({e}); continuing in windowed mode.")
                except Exception:
                    print(f"Fullscreen not available ({e}); continuing in windowed mode.")

        # Reset environment
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        start_time = time.time()
        pause_start_time = 0.0
        total_pause_time = 0.0
        step_count = 0
        last_print_time = start_time
        print_interval = 1.0  # Print gait params every 1 second

        # Tracking for statistics
        gait_param_history = {
            "step_height": [],
            "step_length": [],
            "cycle_time": [],
            "body_height": [],
        }

        # Tracking for trajectory (if requested)
        trajectory_data = []

        was_paused = False

        while True:
            # Calculate elapsed time excluding paused periods
            current_time = time.time()
            if paused["state"]:
                if not was_paused:
                    # Just entered pause state
                    pause_start_time = current_time
                    was_paused = True
                # While paused, freeze elapsed time at the moment we paused
                elapsed = pause_start_time - start_time - total_pause_time
            else:
                if was_paused:
                    # Just exited pause state
                    total_pause_time += current_time - pause_start_time
                    was_paused = False
                # When running, count time excluding all pause periods
                elapsed = current_time - start_time - total_pause_time

            if elapsed >= args.seconds:
                break

            # Handle pause state - skip simulation steps but keep viewer responsive
            if paused["state"]:
                viewer.sync()
                time.sleep(0.01)  # Small sleep to prevent busy-waiting
                continue

            # Store pre-step diagnostics for termination analysis
            if use_vec_env:
                try:
                    sensor_reader = vec_env.get_attr("sensor_reader")[0]
                    pre_quat = sensor_reader.get_body_quaternion()
                    pre_body_pos = sensor_reader.read_sensor("body_pos")
                    pre_linvel = sensor_reader.read_sensor("body_linvel")
                except Exception:
                    pre_quat = None
                    pre_body_pos = None
                    pre_linvel = None
            else:
                try:
                    pre_quat = env.sensor_reader.get_body_quaternion()
                    pre_body_pos = env.sensor_reader.read_sensor("body_pos")
                    pre_linvel = env.sensor_reader.read_sensor("body_linvel")
                except Exception:
                    pre_quat = None
                    pre_body_pos = None
                    pre_linvel = None

            # Get action from policy (or use zero actions for baseline)
            if model is None:
                # Baseline mode: zero actions (pure gait controller)
                if use_vec_env:
                    action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
                else:
                    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            elif use_vec_env:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            # Step environment
            if use_vec_env:
                obs, reward, done, info = vec_env.step(action)
                # Extract info from vec env
                if done[0]:
                    # Check termination reason using pre-step state
                    from envs.adaptive_gait_env import quat_to_euler
                    import math

                    steps = vec_env.get_attr("step_count")[0]
                    print(f"\n{'='*60}")
                    print(f"[t={elapsed:.1f}s] EPISODE TERMINATED after {steps} steps")
                    print(f"{'='*60}")

                    # Try to get orientation from pre-step state
                    if pre_quat is not None and np.linalg.norm(pre_quat) > 1e-6:
                        try:
                            roll, pitch, yaw = quat_to_euler(pre_quat, False)
                            print(f"  Orientation (pre-step):")
                            print(f"    Roll:  {math.degrees(roll):>7.2f}° (limit: ±60°)")
                            print(f"    Pitch: {math.degrees(pitch):>7.2f}° (limit: ±60°)")
                            print(f"    Yaw:   {math.degrees(yaw):>7.2f}°")

                            if abs(roll) > math.pi / 3 or abs(pitch) > math.pi / 3:
                                print(f"  Termination reason: ROBOT TIPPED OVER")
                                if abs(roll) > math.pi / 3:
                                    print(f"    → Roll exceeded limit ({math.degrees(roll):.1f}° > 60°)")
                                if abs(pitch) > math.pi / 3:
                                    print(f"    → Pitch exceeded limit ({math.degrees(pitch):.1f}° > 60°)")
                            else:
                                print(f"  Termination reason: MAX STEPS REACHED")
                        except Exception as e:
                            print(f"  Could not compute Euler angles: {e}")
                    else:
                        print(f"  Termination reason: UNKNOWN (quaternion invalid)")

                    # Additional diagnostics
                    if pre_body_pos is not None:
                        print(f"  Body position: x={pre_body_pos[0]:.3f}m, y={pre_body_pos[1]:.3f}m, z={pre_body_pos[2]:.3f}m")
                    if pre_linvel is not None:
                        print(f"  Linear velocity: x={pre_linvel[0]:.3f}m/s, y={pre_linvel[1]:.3f}m/s, z={pre_linvel[2]:.3f}m/s")

                    # Get current gait params from last info
                    current_info = info[0] if isinstance(info, (list, tuple)) else info
                    if "gait_params" in current_info:
                        params = current_info["gait_params"]
                        print(f"  Active gait params:")
                        print(f"    step_height={params['step_height']:.4f}m, step_length={params['step_length']:.4f}m")
                        print(f"    cycle_time={params['cycle_time']:.3f}s, body_height={params['body_height']:.4f}m")

                    print(f"{'='*60}\n")

                    if not args.no_reset:
                        print(f"  Resetting environment...")
                        obs = vec_env.reset()
                    else:
                        print(f"  Continuing without reset (--no-reset enabled)")

                # Get info from the wrapped info dict
                current_info = info[0] if isinstance(info, (list, tuple)) else info
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if done:
                    from envs.adaptive_gait_env import quat_to_euler
                    import math

                    print(f"\n{'='*60}")
                    print(f"[t={elapsed:.1f}s] EPISODE TERMINATED")
                    print(f"{'='*60}")

                    if pre_quat is not None and np.linalg.norm(pre_quat) > 1e-6:
                        try:
                            roll, pitch, yaw = quat_to_euler(pre_quat, False)
                            print(f"  Orientation (pre-step):")
                            print(f"    Roll:  {math.degrees(roll):>7.2f}° (limit: ±60°)")
                            print(f"    Pitch: {math.degrees(pitch):>7.2f}° (limit: ±60°)")
                            print(f"    Yaw:   {math.degrees(yaw):>7.2f}°")
                        except Exception as e:
                            print(f"  Could not compute Euler angles: {e}")

                    if terminated:
                        print(f"  Termination reason: ROBOT TIPPED OVER")
                    if truncated:
                        print(f"  Termination reason: MAX STEPS REACHED")

                    if pre_body_pos is not None:
                        print(f"  Body position: x={pre_body_pos[0]:.3f}m, y={pre_body_pos[1]:.3f}m, z={pre_body_pos[2]:.3f}m")
                    if pre_linvel is not None:
                        print(f"  Linear velocity: x={pre_linvel[0]:.3f}m/s, y={pre_linvel[1]:.3f}m/s, z={pre_linvel[2]:.3f}m/s")

                    print(f"{'='*60}\n")

                    if not args.no_reset:
                        print(f"  Resetting environment...")
                        obs, _ = env.reset()
                    else:
                        print(f"  Continuing without reset (--no-reset enabled)")
                current_info = info

            # Record trajectory data if requested
            if args.save_trajectory:
                try:
                    if use_vec_env:
                        sensor_reader = vec_env.get_attr("sensor_reader")[0]
                        body_pos = sensor_reader.read_sensor("body_pos")
                    else:
                        body_pos = env.sensor_reader.read_sensor("body_pos")

                    trajectory_data.append({
                        "time": elapsed,
                        "x": float(body_pos[0]),
                        "y": float(body_pos[1]),
                        "z": float(body_pos[2])
                    })
                except Exception as e:
                    pass  # Silently skip if position read fails

            # Update viewer
            viewer.sync()

            # Print gait parameters periodically
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                if "gait_params" in current_info:
                    params = current_info["gait_params"]
                    print(f"[t={elapsed:.1f}s] Gait params: "
                          f"step_h={params['step_height']:.4f}m, "
                          f"step_l={params['step_length']:.4f}m, "
                          f"cycle_t={params['cycle_time']:.3f}s, "
                          f"body_h={params['body_height']:.4f}m")

                    # Track history
                    for key in gait_param_history:
                        gait_param_history[key].append(params[key])

                last_print_time = current_time

            step_count += 1

            # Control playback speed
            time.sleep(env.model.opt.timestep)

        # Get final robot position before closing viewer
        try:
            if use_vec_env:
                sensor_reader = vec_env.get_attr("sensor_reader")[0]
                final_body_pos = sensor_reader.read_sensor("body_pos")
                final_quat = sensor_reader.get_body_quaternion()
                final_linvel = sensor_reader.read_sensor("body_linvel")
            else:
                final_body_pos = env.sensor_reader.read_sensor("body_pos")
                final_quat = env.sensor_reader.get_body_quaternion()
                final_linvel = env.sensor_reader.read_sensor("body_linvel")
        except Exception as e:
            final_body_pos = None
            final_quat = None
            final_linvel = None
            print(f"Warning: Could not read final robot state: {e}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Playback complete!")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Duration:    {elapsed:.1f}s")
    print()

    # Print final robot position
    if final_body_pos is not None:
        print("Final Robot State:")
        print("-" * 40)
        print(f"Position:        x={final_body_pos[0]:.3f}m, y={final_body_pos[1]:.3f}m, z={final_body_pos[2]:.3f}m")

        if final_linvel is not None:
            print(f"Linear velocity: x={final_linvel[0]:.3f}m/s, y={final_linvel[1]:.3f}m/s, z={final_linvel[2]:.3f}m/s")

        if final_quat is not None and np.linalg.norm(final_quat) > 1e-6:
            try:
                from envs.adaptive_gait_env import quat_to_euler
                import math
                roll, pitch, yaw = quat_to_euler(final_quat, False)
                print(f"Orientation:     roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°")
            except Exception:
                pass
        print()

    if gait_param_history["step_height"]:
        print("Gait Parameter Statistics:")
        print("-" * 40)
        for param_name, values in gait_param_history.items():
            arr = np.array(values)
            print(f"{param_name:15s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}")
        print()
        print("Interpretation:")
        print("  - Large std indicates adaptive behavior (good for rough terrain)")
        print("  - Small std indicates policy relies mostly on base parameters")
        print("  - Compare with base gait params to see adaptation magnitude")

    # Save trajectory data if requested
    if args.save_trajectory and trajectory_data:
        output_path = Path(args.save_trajectory)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "mode": "baseline" if args.baseline else "adaptive",
                "duration": elapsed,
                "total_steps": step_count,
                "trajectory": trajectory_data
            }, f, indent=2)

        print(f"Trajectory data saved to: {output_path}")
        print(f"  Total data points: {len(trajectory_data)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
