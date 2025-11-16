#!/usr/bin/env python3
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Int32, Float32MultiArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from gait_controller import DiagonalGaitController, GaitParameters
from ik import solve_leg_ik_3dof

IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0  # +1 keeps controller +X, -1 flips to match leg IK frame


@dataclass(frozen=True)
class LegControl:
    indices: Tuple[int, int, int]
    sign: float
    offset: float


LEG_CONTROL: Dict[str, LegControl] = {
    "FL": LegControl(indices=(0, 1, 2), sign=-1.0, offset=-np.pi),
    "FR": LegControl(indices=(6, 7, 8), sign=1.0, offset=np.pi),
    "RL": LegControl(indices=(3, 4, 5), sign=-1.0, offset=-np.pi),
    "RR": LegControl(indices=(9, 10, 11), sign=1.0, offset=np.pi),
}

GAIT_PARAMS = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)


class RobotControlNode(Node):
    """ROS2 node for publishing camera images and receiving movement commands."""

    def __init__(self):
        super().__init__('robot_control_node')

        # Publishers
        self.camera_publisher = self.create_publisher(RosImage, 'robot_camera', 10)
        self.body_state_publisher = self.create_publisher(Float32MultiArray, 'body_state', 10)

        # Subscribers
        self.movement_subscriber = self.create_subscription(
            Int32,
            'movement_command',
            self.movement_callback,
            10
        )

        # Service server for restarting simulation
        self.restart_service = self.create_service(
            Trigger,
            'restart_simulation',
            self.restart_callback
        )

        # CvBridge for converting images
        self.bridge = CvBridge()

        # Movement command state (0=no movement, 1=up, 2=down)
        self.movement_command = 0

        # Restart simulation flag
        self.restart_requested = False

        self.get_logger().info('Robot Control Node initialized')

    def movement_callback(self, msg):
        """Handle incoming movement commands."""
        self.movement_command = msg.data
        if self.movement_command == 1:
            self.get_logger().info('Movement command: UP')
        elif self.movement_command == 2:
            self.get_logger().info('Movement command: DOWN')
        elif self.movement_command == 0:
            self.get_logger().info('Movement command: STOP')

    def restart_callback(self, request, response):
        """Handle simulation restart requests."""
        self.get_logger().info('Restart simulation service called')
        self.restart_requested = True
        response.success = True
        response.message = 'Simulation restart requested'
        return response

    def publish_camera_image(self, pixels):
        """Publish camera image to ROS2 topic."""
        try:
            # Convert numpy array (RGB) to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(pixels, encoding='rgb8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'robot_camera'
            self.camera_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

    def publish_body_state(self, position, orientation_quat):
        """Publish body position and orientation (as Euler angles) to ROS2 topic.

        Args:
            position: (x, y, z) position in meters
            orientation_quat: (w, x, y, z) quaternion
        """
        try:
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            rotation = Rotation.from_quat([orientation_quat[1], orientation_quat[2],
                                          orientation_quat[3], orientation_quat[0]])
            euler = rotation.as_euler('xyz', degrees=False)

            # Create message: [x, y, z, roll, pitch, yaw]
            msg = Float32MultiArray()
            msg.data = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(euler[0]),  # roll
                float(euler[1]),  # pitch
                float(euler[2])   # yaw
            ]

            self.body_state_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish body state: {e}')


# Global variables for MuJoCo model and data (initialized in main)
model = None
data = None
robot_body_id = None
body_pos_sensor_id = None
body_quat_sensor_id = None


def apply_leg_angles(ctrl: mujoco.MjData, leg: str, angles: Tuple[float, float, float]) -> None:
    """Map IK output angles into the actuator ordering."""
    tilt, ang_left, ang_right = angles
    config = LEG_CONTROL[leg]
    idx_left, idx_right, idx_tilt = config.indices
    sign = config.sign
    offset = config.offset

    ctrl.ctrl[idx_left] = sign * ang_left
    ctrl.ctrl[idx_right] = sign * ang_right + offset
    ctrl.ctrl[idx_tilt] = tilt


def apply_gait_targets(controller: DiagonalGaitController, timestep: float, movement_command: int = 0) -> None:
    """Evaluate the gait planner and push the resulting joint targets to MuJoCo.

    Args:
        controller: The gait controller
        timestep: Time step for simulation
        movement_command: 0=stop, 1=forward, 2=backward
    """
    # Adjust gait based on movement command
    if movement_command == 0:
        # No movement - freeze gait (don't update time)
        timestep = 0.0
    elif movement_command == 2:
        # Down/backward - invert the forward direction
        timestep = timestep  # Normal timestep, but we'll flip the direction

    leg_targets = controller.update(timestep)

    for leg in LEG_CONTROL:
        target = leg_targets.get(leg)
        if target is None:
            continue

        # Map controller forward direction to leg-local IK frame.
        # Current robot geometry yields opposite X sense; flip to move forward.
        target_local = target.copy()

        # Apply direction based on movement command
        if movement_command == 2:
            # Invert X direction for backward movement
            target_local[0] *= -FORWARD_SIGN
        else:
            target_local[0] *= FORWARD_SIGN

        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is None:
            print(f"[WARN] IK failed for leg {leg} with target {target}")
            continue

        apply_leg_angles(data, leg, result)


def main() -> None:
    global model, data, robot_body_id, body_pos_sensor_id, body_quat_sensor_id

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Quadruped robot simulation with MuJoCo and ROS2')
    parser.add_argument('--terrain', type=str, choices=['flat', 'rough'], default='flat',
                        help='Terrain type: flat (default) or rough')
    args = parser.parse_args()

    # Select world file based on terrain argument
    world_file = "model/world.xml" if args.terrain == 'flat' else "model/world_train.xml"
    print(f"Loading world: {world_file} ({args.terrain} terrain)")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(world_file)
    data = mujoco.MjData(model)
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    # Get sensor IDs
    body_pos_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_pos")
    body_quat_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat")

    # Initialize ROS2
    rclpy.init()

    # Create ROS2 node
    ros_node = RobotControlNode()

    controller = DiagonalGaitController(GAIT_PARAMS)
    controller.reset()

    # Create offscreen renderer for camera captures
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera")

    # Track time for periodic image capture (every 0.1 seconds)
    last_capture_time = 0.0
    capture_interval = 0.1  # seconds

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Process ROS2 callbacks
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # Check if restart was requested
            if ros_node.restart_requested:
                ros_node.get_logger().info('Restarting MuJoCo simulation...')
                # Reset simulation data in-place
                mujoco.mj_resetData(model, data)
                # Reset gait controller
                controller.reset()
                # Reset camera capture timer (data.time gets reset to 0)
                last_capture_time = 0.0
                # Clear restart flag
                ros_node.restart_requested = False
                ros_node.get_logger().info('Simulation restarted successfully')

            # Apply gait with movement command from ROS2
            apply_gait_targets(controller, model.opt.timestep, ros_node.movement_command)
            mujoco.mj_step(model, data)

            robot_pos = data.xpos[robot_body_id]
            viewer.cam.lookat[:] = robot_pos
            viewer.sync()

            # Capture and publish image every 0.1 seconds
            current_time = data.time
            if current_time - last_capture_time >= capture_interval:
                # Render from robot camera
                renderer.update_scene(data, camera=camera_id)
                pixels = renderer.render()

                # Publish to ROS2 topic
                ros_node.publish_camera_image(pixels)

                # Read body sensors and publish body state
                body_pos_adr = model.sensor_adr[body_pos_sensor_id]
                body_quat_adr = model.sensor_adr[body_quat_sensor_id]

                body_position = data.sensordata[body_pos_adr:body_pos_adr + 3]
                body_orientation = data.sensordata[body_quat_adr:body_quat_adr + 4]

                ros_node.publish_body_state(body_position, body_orientation)

                last_capture_time = current_time

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    # Cleanup ROS2
    ros_node.destroy_node()
    rclpy.shutdown()
    print("Demo finished")


if __name__ == "__main__":
    main()
