#!/usr/bin/env python3
"""Diagnose lateral movement over multiple gait cycles."""

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand
import numpy as np

# Initialize gait controller
params = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
controller = DiagonalGaitController(params, robot_width=0.1)
controller.reset()

print("=== Lateral Movement Over Multiple Cycles ===\n")

# Set lateral velocity
vel = VelocityCommand(vx=0.0, vy=0.1, omega=0.0)

# Simulate multiple cycles
dt = 0.02  # 20ms timestep
num_steps = 200  # 4 seconds
cycle_duration = params.cycle_time

print(f"Simulating {num_steps * dt:.1f} seconds of lateral motion (vy=0.1 m/s)")
print(f"Cycle time: {cycle_duration:.2f}s\n")

time = 0.0
cycle_count = 0
last_state = controller.state

# Track leg positions over time
fl_y_history = []
fr_y_history = []

for step in range(num_steps):
    targets = controller.update(dt, vel)
    time += dt

    # Track state changes
    if controller.state != last_state:
        cycle_count += 1
        last_state = controller.state

    # Record Y positions
    fl_y_history.append(targets['FL'][1])
    fr_y_history.append(targets['FR'][1])

    # Print every 0.2 seconds
    if step % 10 == 0:
        print(f"t={time:4.2f}s  State={controller.state:15s}  "
              f"FL_y={targets['FL'][1]:7.4f}  FR_y={targets['FR'][1]:7.4f}  "
              f"RL_y={targets['RL'][1]:7.4f}  RR_y={targets['RR'][1]:7.4f}")

print(f"\nCompleted {cycle_count} half-cycles\n")

# Analyze Y-position trends
fl_y_arr = np.array(fl_y_history)
fr_y_arr = np.array(fr_y_history)

# Check if positions oscillate or trend
fl_trend = fl_y_arr[-20:].mean() - fl_y_arr[:20].mean()
fr_trend = fr_y_arr[-20:].mean() - fr_y_arr[:20].mean()

print("Analysis:")
print(f"  FL Y-position: start={fl_y_arr[:20].mean():.4f}, end={fl_y_arr[-20:].mean():.4f}, trend={fl_trend:+.4f}")
print(f"  FR Y-position: start={fr_y_arr[:20].mean():.4f}, end={fr_y_arr[-20:].mean():.4f}, trend={fr_trend:+.4f}")
print(f"  FL Y-range: {fl_y_arr.min():.4f} to {fl_y_arr.max():.4f} (variation: {fl_y_arr.max()-fl_y_arr.min():.4f})")
print(f"  FR Y-range: {fr_y_arr.min():.4f} to {fr_y_arr.max():.4f} (variation: {fr_y_arr.max()-fr_y_arr.min():.4f})")

print("\n" + "=" * 70)
if abs(fl_trend) < 0.005 and abs(fr_trend) < 0.005:
    print("ISSUE DETECTED: Y-positions oscillate but don't trend!")
    print("The legs are moving back and forth laterally, not translating continuously.")
else:
    print("OK: Y-positions show a trending motion (continuous translation)")
print("=" * 70)
