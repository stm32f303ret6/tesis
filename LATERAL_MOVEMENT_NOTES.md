# Lateral Movement Implementation

## Overview
The quadruped robot now supports full 3-DOF velocity control:
- **vx**: Forward/backward velocity (m/s)
- **vy**: Lateral (left/right) velocity (m/s)
- **omega**: Yaw rotation velocity (rad/s)

## ⚠️ Important: Lateral Gain Factor
Lateral motion in a trot gait requires a **gain factor** to overcome natural body sway. The implementation uses `lateral_gain = 2.2` to amplify lateral steps and achieve ~89% of commanded lateral velocity.

## How Lateral Movement Works

### 1. Gait Controller
The `DiagonalGaitController` computes lateral displacement from velocity:
```python
self.step_y = self.current_velocity.vy * self.state_duration
```

Both swing and stance trajectories include Y-displacement:
- **Swing curve**: Bézier curve includes Y-nodes `[-half_step_y, ..., half_step_y]`
- **Stance path**: Linear sweep includes `y_pos = half_step_y - (step_y * tau)`

### 2. 3DOF Inverse Kinematics
The parallel SCARA leg uses the **tilt joint** to achieve lateral displacement:

```python
# From ik.py - parallel_scara_ik_3dof()
if tilt_angle is None:
    tilt_angle = math.atan2(y, abs(z))  # Compute tilt from desired Y

# Transform target into tilted leg frame
z_eff = z * cos(tilt) + y * sin(tilt)
```

The tilt joint rotates around the X-axis, allowing the leg to reach laterally while maintaining proper foot placement.

### 3. Typical Tilt Angles
From test results with max_linear_vel = 0.15 m/s:
- **Right lateral (vy=0.1)**: Tilt angles ±20-22°
- **Left lateral (vy=-0.1)**: Tilt angles ±20-22° (opposite sign)
- **Combined motion**: Tilt angles adjust proportionally

## Bugs That Were Fixed

### Bug 1: Y-Offset Replacement (Initial Issue)
**Problem**: The `_apply_lateral_offset()` function was **replacing** the Y-component instead of adding to it:
```python
# BEFORE (broken)
adjusted[1] = lateral  # Overwrites Y-displacement!
```

**Solution**: Changed to **add** the offset:
```python
# AFTER (fixed)
adjusted[1] += lateral  # Preserves Y-displacement from gait
```

### Bug 2: Lateral Translation Stops After 2 Seconds (Synchronization Issue)
**Problem**: The robot would move laterally initially but stop after ~2 seconds because:
1. Natural body sway (±0.044m) was **larger** than the lateral step size (0.04m)
2. The oscillation overwhelmed the intended lateral displacement
3. Net lateral velocity approached zero as the robot settled into oscillating in place

**Diagnosis**:
- With `vy=0.1 m/s` and `cycle_time=0.8s`, `step_y = 0.04m` per half-cycle
- Body sway amplitude was ±0.044m, larger than step size!
- Result: oscillation dominated, no net translation

**Solution**: Added `lateral_gain = 2.2` to amplify lateral steps:
```python
# gait_controller.py line 110
lateral_gain = 2.2
self.step_y = self.current_velocity.vy * self.state_duration * lateral_gain
```

**Results**:
- Achieves ~89% of commanded lateral velocity
- Continuous translation (no stopping after 2s)
- Body sway: ±0.23m (larger but controlled)
- Tilt angles: ±24° (well within limits)

## Joystick Controls

### Gamepad
- **Left stick X-axis**: Lateral movement (vy)
- **Left stick Y-axis**: Forward/backward (vx)
- **Right stick X-axis**: Rotation (omega)

### Keyboard Fallback
- **A/D keys**: Lateral left/right
- **W/S keys**: Forward/backward
- **Q/E keys**: Rotate left/right

## Testing

Run these test scripts to verify lateral movement:
```bash
# Test lateral displacement in gait controller
python3 test_lateral_movement.py

# Verify IK can solve lateral targets
python3 test_ik_with_lateral.py

# Run full simulation with joystick
python3 height_control.py
```

## Reachability Notes

The 3DOF leg can reach targets within:
- **X-range**: ±(L1 + L2) = ±0.105 m
- **Y-range**: Limited by tilt joint range and foot clearance
- **Z-range**: 0 to -(L1 + L2) = -0.105 m (downward)

Current velocity limits (max_linear_vel=0.15 m/s) produce step displacements well within the reachable workspace.
