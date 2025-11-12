#!/usr/bin/env python3
# PyQt5: pip install PyQt5
# Pygame: pip install pygame

import sys
import os
import sqlite3
import pygame
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi


class GuiRosNode(Node):
    """ROS2 node for GUI - subscribes to camera images and publishes movement commands."""

    def __init__(self):
        super().__init__('gui_ros_node')

        # Subscribers
        self.camera_subscriber = self.create_subscription(
            RosImage,
            'robot_camera',
            self.camera_callback,
            10
        )

        # Publishers
        self.movement_publisher = self.create_publisher(Int32, 'movement_command', 10)

        # CvBridge for converting images
        self.bridge = CvBridge()

        # Store latest image
        self.latest_image = None

        self.get_logger().info('GUI ROS Node initialized')

    def camera_callback(self, msg):
        """Handle incoming camera images."""
        try:
            # Convert ROS Image to numpy array (RGB)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def publish_movement_command(self, command):
        """Publish movement command (0=no movement, 1=up, 2=down)."""
        msg = Int32()
        msg.data = command
        self.movement_publisher.publish(msg)


class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()

        # Store ROS2 node reference
        self.ros_node = ros_node

        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load UI file
        ui_path = os.path.join(self.script_dir, "untitled.ui")
        loadUi(ui_path, self)

        # Initialize pygame for joystick
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            try:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Joystick detected: {self.joystick.get_name()}")
            except pygame.error as e:
                print(f"Joystick initialization failed: {e}")
                print("Continuing without joystick support")
                self.joystick = None
        else:
            print("No joystick detected")

        # Database path
        self.db_path = os.path.join(self.script_dir, "users.db")

        # User role
        self.user_role = None

        # Setup initial UI state
        self.setup_ui()

        # Connect login button
        self.login_btn.clicked.connect(self.handle_login)

        # Setup timer for joystick polling
        self.joystick_timer = QTimer()
        self.joystick_timer.timeout.connect(self.poll_joystick)
        self.joystick_timer.start(50)  # Poll every 50ms

        # Setup timer for ROS2 spin
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.process_ros)
        self.ros_timer.start(10)  # Process ROS2 callbacks every 10ms

        # Setup timer for camera display update
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_display)
        self.camera_timer.start(100)  # Update display every 100ms (0.1 seconds)

        # Track current joystick state
        self.joystick_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False
        }

        # Track last published movement command
        self.last_movement_command = 0

    def setup_ui(self):
        """Initialize UI elements."""
        # Hide operation tab initially
        self.tabWidget.setTabEnabled(self.tabWidget.indexOf(self.tab_operation), False)
        self.tabWidget.setCurrentWidget(self.tab_login)

        # Hide owner panel initially
        self.owner_panel.setVisible(False)

        # Set default brown images for arrow labels
        self.set_arrow_images()

    def set_arrow_images(self, up='brown', down='brown', left='brown', right='brown'):
        """Set arrow images based on state."""
        icons_dir = os.path.join(self.script_dir, "icons")

        # Maximum size for arrow images (width, height in pixels)
        max_size = 100

        # Set up arrow
        up_pixmap = QPixmap(os.path.join(icons_dir, f"up_{up}.png"))
        if not up_pixmap.isNull():
            self.up_label.setPixmap(up_pixmap.scaled(
                max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # Set down arrow
        down_pixmap = QPixmap(os.path.join(icons_dir, f"down_{down}.png"))
        if not down_pixmap.isNull():
            self.down_label.setPixmap(down_pixmap.scaled(
                max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # Set left arrow (using RIGHT image - swapped)
        left_pixmap = QPixmap(os.path.join(icons_dir, f"right_{left}.png"))
        if not left_pixmap.isNull():
            self.left_label.setPixmap(left_pixmap.scaled(
                max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # Set right arrow (using LEFT image - swapped)
        right_pixmap = QPixmap(os.path.join(icons_dir, f"left_{right}.png"))
        if not right_pixmap.isNull():
            self.right_label.setPixmap(right_pixmap.scaled(
                max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def handle_login(self):
        """Handle login button click."""
        username = self.user_text.text()
        password = self.pass_text.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Por favor ingrese usuario y contraseña")
            return

        # Check credentials against database
        role = self.check_credentials(username, password)

        if role:
            self.user_role = role
            # Login successful
            self.show_operation_tab()
        else:
            # Login failed
            QMessageBox.critical(self, "Error", "Usuario o contraseña incorrectos")
            self.pass_text.clear()

    def check_credentials(self, username, password):
        """Check username and password against database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT role FROM users WHERE username = ? AND password = ?',
                (username, password)
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return result[0]  # Return the role
            else:
                return None
        except Exception as e:
            print(f"Database error: {e}")
            return None

    def show_operation_tab(self):
        """Switch to operation tab and configure based on user role."""
        # Enable and switch to operation tab
        self.tabWidget.setTabEnabled(self.tabWidget.indexOf(self.tab_operation), True)
        self.tabWidget.setCurrentWidget(self.tab_operation)

        # Disable login tab
        self.tabWidget.setTabEnabled(self.tabWidget.indexOf(self.tab_login), False)

        # Show/hide owner panel based on role
        if self.user_role == 'owner':
            self.owner_panel.setVisible(True)
        else:
            self.owner_panel.setVisible(False)

        # Always show left and right panels
        self.left_panel.setVisible(True)
        self.right_panel.setVisible(True)

        print(f"Login successful. Role: {self.user_role}")

    def process_ros(self):
        """Process ROS2 callbacks."""
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)

    def update_camera_display(self):
        """Update camera_label with latest image from ROS2."""
        if self.ros_node.latest_image is not None:
            try:
                # Convert numpy array (RGB) to QImage
                height, width, channel = self.ros_node.latest_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    self.ros_node.latest_image.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888
                )

                # Convert QImage to QPixmap and display
                pixmap = QPixmap.fromImage(q_image)

                # Scale to fit camera_label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                self.camera_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"Error updating camera display: {e}")

    def poll_joystick(self):
        """Poll joystick state and update UI."""
        if not self.joystick:
            return

        # Process pygame events
        pygame.event.pump()

        # Check D-pad/hat (usually hat 0)
        new_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False
        }

        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)

            # Hat returns (x, y) where:
            # x: -1 (left), 0 (center), 1 (right)
            # y: -1 (down), 0 (center), 1 (up)

            if hat[1] == 1:  # Up
                new_state['up'] = True
            elif hat[1] == -1:  # Down
                new_state['down'] = True

            if hat[0] == -1:  # Left
                new_state['left'] = True
            elif hat[0] == 1:  # Right
                new_state['right'] = True

        # Also check axes as backup (some controllers use axes for D-pad)
        if self.joystick.get_numaxes() >= 2:
            axis_x = self.joystick.get_axis(0)  # Left stick X
            axis_y = self.joystick.get_axis(1)  # Left stick Y

            threshold = 0.5

            if axis_y < -threshold:  # Up (Y is inverted)
                new_state['up'] = True
            elif axis_y > threshold:  # Down
                new_state['down'] = True

            if axis_x < -threshold:  # Left
                new_state['left'] = True
            elif axis_x > threshold:  # Right
                new_state['right'] = True

        # Update images if state changed
        if new_state != self.joystick_state:
            self.joystick_state = new_state
            self.set_arrow_images(
                up='orange' if new_state['up'] else 'brown',
                down='orange' if new_state['down'] else 'brown',
                left='orange' if new_state['left'] else 'brown',
                right='orange' if new_state['right'] else 'brown'
            )

        # Publish movement command via ROS2
        # 0 = no movement, 1 = up, 2 = down
        movement_command = 0
        if new_state['up']:
            movement_command = 1
        elif new_state['down']:
            movement_command = 2

        # Only publish if command changed
        if movement_command != self.last_movement_command:
            self.ros_node.publish_movement_command(movement_command)
            self.last_movement_command = movement_command

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self.joystick_timer.stop()
        self.ros_timer.stop()
        self.camera_timer.stop()
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        event.accept()


def main():
    # Initialize ROS2
    rclpy.init()

    # Create ROS2 node
    ros_node = GuiRosNode()

    # Create Qt application and main window
    app = QApplication(sys.argv)
    w = MainWindow(ros_node)
    w.show()

    # Run application
    exit_code = app.exec_()

    # Cleanup ROS2
    ros_node.destroy_node()
    rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
