# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2-integrated quadruped robot simulation using the MuJoCo physics engine. The robot features a novel 3DOF parallel SCARA leg design with tilt control, allowing independent control of leg height, lateral position, and tilt angle. The system includes a PyQt5 GUI with joystick support for teleoperation, as well as standalone simulation capabilities.

## Core Architecture

### Inverse Kinematics System (`ik.py`)
The IK module provides the mathematical foundation for leg control:
- `solve_2link_ik()`: Basic 2-link planar IK solver using law of cosines
- `parallel_scara_ik()`: 2DOF parallel SCARA mechanism with 4 working modes (different elbow configurations)
- `parallel_scara_ik_3dof()`: Full 3DOF IK including tilt control for 3D target positions
- `solve_leg_ik_3dof()`: Convenience wrapper with robot-specific default parameters

**Key parameters:**
- `L1 = 0.045` m (upper link length)
- `L2 = 0.06` m (lower link length)
- `base_dist = 0.021` m (distance between parallel arms)
- `mode`: 1=A up/B down, 2=A down/B up (default), 3=both up, 4=both down

The 3DOF IK transforms 3D targets (x, y, z) into joint angles (tilt, shoulder_L, shoulder_R) by:
1. Computing or accepting a tilt angle to handle y-displacement
2. Projecting the target into the tilted leg's planar frame
3. Solving the 2DOF SCARA IK in that plane

### Gait Controller (`gait_controller.py`)
State machine-based diagonal gait generator using `transitions` library for coordination and `bezier` for smooth swing trajectories:
- `DiagonalGaitController`: Main controller class managing trot gait with diagonal leg pairs
- `GaitParameters`: Dataclass for gait configuration (body_height, step_length, step_height, cycle_time, swing_shape, lateral_offsets)
- **Diagonal pairs:** FL+RR (pair_a), FR+RL (pair_b) alternate between swing and stance
- **State machine:** Two states (`pair_a_swing`, `pair_b_swing`) with automatic transitions every half-cycle
- **Swing trajectory:** Cubic Bézier curve with configurable shape parameter for lift/touchdown dynamics
- **Stance trajectory:** Linear sweep from front to rear over the stance duration

The controller outputs per-leg foot targets in leg-local frame, which are then transformed and fed to IK.

### Simulation Scripts

#### Standalone Simulation (`height_control.py`)
Standalone entry point for running the robot simulation without ROS2:
- Loads robot model from `model/world_train.xml` (rough terrain) or `model/world.xml` (flat)
- `LEG_CONTROL` dictionary: Maps leg names to actuator indices, signs, and offsets
- `apply_leg_angles()`: Transforms IK output (tilt, shoulder_L, shoulder_R) into MuJoCo actuator commands
- `apply_gait_targets()`: Main control loop that evaluates gait, solves IK, applies commands
- `FORWARD_SIGN = -1.0`: Flips controller +X to match leg IK frame for forward motion
- **Control mapping:** FL=(0,1,2), RL=(3,4,5), FR=(6,7,8), RR=(9,10,11); rear shoulders negated, front/right shoulders offset by π

#### ROS2-Integrated Simulation (`sim.py`)
ROS2-integrated MuJoCo simulation with GUI communication:
- **Node name:** `robot_control_node` (class: `RobotControlNode`)
- Loads robot model from `model/world.xml` (flat) or `model/world_train.xml` (rough)
- Uses `DiagonalGaitController` for autonomous walking patterns
- `apply_gait_targets()`: Evaluates gait planner and pushes joint targets to MuJoCo
- **Control mapping:** FL(0-2), RL(3-5), FR(6-8), RR(9-11); each leg has [left_shoulder, right_shoulder, tilt]
- Camera follows robot body during simulation
- **FORWARD_SIGN = -1.0**: Flips controller X-axis to match leg IK frame orientation

### ROS2 Integration Layer
The system uses ROS2 for inter-process communication between simulation and GUI:

**Topics:**
- `/robot_camera` (sensor_msgs/Image): RGB camera feed at 10 Hz (640x480, published by sim.py)
- `/movement_command` (std_msgs/Int32): Joystick commands (published by gui.py)
  - 0 = stop (freezes gait), 1 = forward, 2 = backward
- `/body_state` (std_msgs/Float32MultiArray): Robot pose [x, y, z, roll, pitch, yaw] in meters and radians

**Services:**
- `/restart_simulation` (std_srvs/Trigger): Resets MuJoCo simulation and gait controller (called from GUI X button)

**Sensors in MuJoCo model:**
- `body_pos`: Framepos sensor for robot body position
- `body_quat`: Framequat sensor for robot body orientation
- `robot_camera`: Camera mounted on robot body for first-person view

### MuJoCo Model Structure
- `model/world.xml`: Flat ground plane scene with checker texture
- `model/world_train.xml`: Rough terrain scene using heightfield from `hfield.png` for training/testing
- `model/robot.xml`: Complete robot definition with 4 legs, each having:
  - Tilt joint (1 DOF, axis=[1,0,0])
  - Two shoulder joints (parallel SCARA, axis=[0,-1,0])
  - Two elbow joints (passive/constrained)
- `model/assets/`: STL mesh files for all robot parts
- `model/openscad/`: Editable CAD sources (`.scad`) and generated `.stl` files
- `model/primitives_model/`: Simplified box geometry model for debugging
- `model/world_static_test.xml`, `model/robot_static_test.xml`: Static test variants

**Important:** Joint names and body names in XML must stay stable to avoid breaking control code that references them by name.

### Utility Scripts
- `foot_range_calculator.py`: Calculates reachable workspace for each foot using IK, validates safe parameter ranges
- `tests/compare_world_trajectories.py`: Compares robot trajectories between flat and rough terrain, generates matplotlib plots

## Development Commands

### Environment Setup
```bash
# Source ROS2 environment (required for ROS2 operations)
source /opt/ros/jazzy/setup.bash

# Install Python dependencies
pip install -r requirements.txt
# Includes: mujoco, numpy, scipy, transitions, bezier, matplotlib
# ROS2 dependencies: rclpy, sensor_msgs, std_msgs, std_srvs, cv_bridge
```

### Running the Full System (ROS2)
```bash
# Terminal 1: Start MuJoCo simulation with ROS2 integration
python3 sim.py                    # Flat terrain (default)
python3 sim.py --terrain flat     # Flat terrain (explicit)
python3 sim.py --terrain rough    # Rough terrain for testing

# Terminal 2: Start PyQt5 GUI with joystick control
cd gui
python3 gui.py
```

The simulation publishes camera images to `/robot_camera` at 10 Hz and listens for movement commands on `/movement_command`. The GUI displays the camera feed and sends joystick inputs.

### Running Standalone (No ROS2)
```bash
# Main gait controller demo on rough terrain (default)
python3 height_control.py

# Headless mode (for CI/testing, requires MUJOCO_GL=egl)
MUJOCO_GL=egl timeout 60 python3 height_control.py
```

### Running Standalone Scripts
```bash
# IK verification tests (standalone, no ROS2)
python3 ik.py

# Gait controller verification (standalone, no ROS2)
python3 gait_controller.py

# Calculate foot workspace and validate safe ranges
python3 foot_range_calculator.py

# Compare trajectories between flat and rough terrain
python3 tests/compare_world_trajectories.py

# Headless simulation mode (for CI/testing)
MUJOCO_GL=egl timeout 10 python3 sim.py
```

### ROS2 Debugging and Monitoring
```bash
# List active topics
ros2 topic list

# Monitor camera feed rate
ros2 topic hz /robot_camera

# Echo movement commands
ros2 topic echo /movement_command

# View body state (position + orientation)
ros2 topic echo /body_state

# List active nodes (should show robot_control_node and gui_ros_node)
ros2 node list

# Manually publish movement commands for testing
ros2 topic pub /movement_command std_msgs/Int32 "data: 1"  # Forward
ros2 topic pub /movement_command std_msgs/Int32 "data: 0"  # Stop

# Restart simulation via service
ros2 service call /restart_simulation std_srvs/Trigger
```

### Asset Pipeline
When modifying robot geometry:
1. Edit `.scad` files in `model/openscad/`
2. Regenerate STL: `openscad -o model/assets/<part>.stl model/openscad/<part>.scad`
3. Verify mesh orientation and scale in MuJoCo before committing
4. Update link lengths in code if dimensions changed

## Coding Conventions

### Python Style
- Follow PEP 8: 4-space indentation, line length ≤ 100
- Module constants in UPPER_SNAKE_CASE, functions/variables in snake_case
- Group imports: standard library, third-party (numpy, mujoco, rclpy), local modules
- Keep docstrings focused on coordinate frames and units (meters, radians)

### Testing Approach
- `ik.py` and `gait_controller.py` contain built-in verification when run as main scripts
- For new features, add pytest-compatible tests in `tests/test_*.py`
- Before committing kinematic or gait changes, visually verify with `python3 height_control.py` or `python3 sim.py`
- Check console for MuJoCo warnings about joint limits or contacts
- Test ROS2 integration by running both `sim.py` and `gui/gui.py` simultaneously

## Common Pitfalls

1. **Coordinate frame confusion**: The IK operates in leg-local frame where Z points down (gravity direction). Target positions should be negative Z for downward reach. The gait controller uses forward +X, but this is flipped via `FORWARD_SIGN = -1.0` to match leg IK frame.

2. **Control index mapping**: FL=(0,1,2), RL=(3,4,5), FR=(6,7,8), RR=(9,10,11). Each leg has 3 actuators in order: left shoulder, right shoulder, tilt. The LEG_CONTROL dictionary in both `height_control.py` and `sim.py` defines the mapping with appropriate signs and offsets.

3. **Shoulder angle transformations**: Rear legs (FL, RL) have shoulders negated (`sign=-1.0`), front/right legs (FR, RR) have right shoulder offset by π. See `LEG_CONTROL` dictionary in height_control.py:24-29 or sim.py.

4. **ROS2 environment**: Always source `/opt/ros/jazzy/setup.bash` before running ROS2-integrated scripts (`sim.py`, `gui/gui.py`). Without it, you'll get import errors for `rclpy`.

5. **Movement command semantics** (ROS2 mode):
   - 0 = stop (sets timestep to 0, freezing gait in place)
   - 1 = forward (normal gait)
   - 2 = backward (inverts X direction by multiplying by -FORWARD_SIGN)

6. **Asset regeneration**: When updating SCAD files, remember to regenerate STL and copy to `model/assets/` before testing.

7. **Reachability limits**: Max reach = L1 + L2 = 0.105m. Practical working range validated by `foot_range_calculator.py`.

8. **Gait parameters**: When tuning `GaitParameters`, ensure step_height + body_height doesn't exceed reachable workspace. Use foot_range_calculator.py to verify targets are reachable before testing in simulation.

9. **State machine timing**: The gait controller auto-transitions every `state_duration = cycle_time / 2.0`. Ensure MuJoCo timestep is small enough to capture smooth trajectory updates (default 0.002s works well).

## Key Dependencies and Their Roles

### Core Simulation
- **mujoco**: Physics engine and rendering
- **numpy**: Numerical operations for IK and trajectory generation

### Gait Generation
- **transitions**: State machine library for gait phase management (pair_a_swing ↔ pair_b_swing)
- **bezier**: Cubic Bézier curve library for smooth swing trajectories

### ROS2 Communication
- **rclpy**: ROS2 Python client library
- **sensor_msgs**: Image message type for camera feed
- **std_msgs**: Int32 for movement commands, Float32MultiArray for body state
- **std_srvs**: Trigger service for simulation restart
- **cv_bridge**: Converts between ROS Image messages and numpy/OpenCV arrays

### GUI (gui/ directory)
- **PyQt5**: GUI framework
- **joystick support**: Reads gamepad inputs for teleoperation

### Analysis Tools
- **scipy**: Quaternion to Euler angle conversion (Rotation class)
- **matplotlib**: Plotting foot trajectories and analysis results (foot_range_calculator.py)

## Project Documentation

- **ROS2_INTEGRATION.md**: Detailed ROS2 setup, topic descriptions, and troubleshooting
- **ros2_topology.puml**: PlantUML diagram showing ROS2 node/topic architecture
- **gait_controller_plan.md**: Design document for gait controller implementation
- **foot_range_results.txt**: Analysis of foot workspace reachability

## Commit Style

Match repository history with brief, present-tense gerund subjects (e.g., "adding topology puml and service", "fixing plot", "adding ros2 support"). Reference design docs in body when relevant. Note MuJoCo model changes or ROS2 API changes in commit messages.
