# ROS2 Integration Guide

## Overview

The quadruped robot simulation is now integrated with ROS2 for communication between the MuJoCo simulation (`height_control.py`) and the GUI (`gui/gui.py`).

## ROS2 Topics

### `/robot_camera` (sensor_msgs/Image)
- **Publisher**: `height_control.py` (robot_control_node)
- **Subscriber**: `gui/gui.py` (gui_ros_node)
- **Rate**: 10 Hz (every 0.1 seconds)
- **Description**: Camera images from the robot's onboard camera in the MuJoCo simulation

### `/movement_command` (std_msgs/Int32)
- **Publisher**: `gui/gui.py` (gui_ros_node)
- **Subscriber**: `height_control.py` (robot_control_node)
- **Description**: Movement commands from joystick
  - `0`: Stop - robot freezes in place (gait halted)
  - `1`: Forward - robot walks forward
  - `2`: Backward - robot walks backward (inverted direction)
  - (Left/right to be implemented later)

## Running the System

### Terminal 1 - Start the Simulation
```bash
cd /home/rsc/Desktop/repos/joy2
python3 height_control.py
```

This will:
- Start the MuJoCo simulation with interactive viewer
- Publish camera images to `/robot_camera` topic every 0.1 seconds
- Listen for movement commands on `/movement_command` topic
- Display movement commands in the console

### Terminal 2 - Start the GUI
```bash
cd /home/rsc/Desktop/repos/joy2/gui
python3 gui.py
```

This will:
- Start the PyQt5 GUI with joystick support
- Display camera images from `/robot_camera` in the camera_label
- Publish joystick commands to `/movement_command` topic
- Update camera display every 0.1 seconds

## Login Credentials

Use the existing database credentials to log in and access the operation tab where the camera feed will be displayed.

## Monitoring ROS2 Topics

Check if topics are active:
```bash
ros2 topic list
```

Monitor camera images:
```bash
ros2 topic hz /robot_camera
```

Monitor movement commands:
```bash
ros2 topic echo /movement_command
```

View topic info:
```bash
ros2 topic info /robot_camera
ros2 topic info /movement_command
```

## Testing Without Joystick

You can manually publish movement commands for testing:
```bash
# Stop (robot freezes in place)
ros2 topic pub /movement_command std_msgs/Int32 "data: 0"

# Move forward (robot walks forward)
ros2 topic pub /movement_command std_msgs/Int32 "data: 1"

# Move backward (robot walks in reverse)
ros2 topic pub /movement_command std_msgs/Int32 "data: 2"
```

## Troubleshooting

### No camera image in GUI
1. Check if `height_control.py` is running and publishing:
   ```bash
   ros2 topic hz /robot_camera
   ```
2. Check ROS2 node list:
   ```bash
   ros2 node list
   ```
   Should show: `/robot_control_node` and `/gui_ros_node`

### Movement commands not received
1. Check if GUI is publishing:
   ```bash
   ros2 topic echo /movement_command
   ```
2. Press joystick up/down and verify commands are published

### ROS2 initialization errors
Make sure ROS2 is properly sourced:
```bash
source /opt/ros/jazzy/setup.bash
```

## Implementation Details

### height_control.py
- **Node name**: `robot_control_node`
- **Class**: `RobotControlNode`
- Uses `cv_bridge` to convert numpy arrays to ROS Image messages
- Publishes RGB images (640x480) at 10 Hz
- Logs movement commands when received

### gui/gui.py
- **Node name**: `gui_ros_node`
- **Class**: `GuiRosNode`
- Uses `cv_bridge` to convert ROS Image messages to numpy arrays
- Converts numpy arrays to QPixmap for display in camera_label
- Publishes movement commands only when joystick state changes
- Three timers:
  - Joystick polling: 50 ms
  - ROS2 spin: 10 ms
  - Camera display update: 100 ms

## Next Steps

- [ ] Implement left/right movement commands (values 3 and 4)
- [ ] Add camera controls (zoom, pan, etc.)
- [ ] Integrate movement commands with gait controller
- [ ] Add telemetry data publishing (robot pose, velocity, etc.)
