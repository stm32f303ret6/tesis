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
from std_msgs.msg import Int32, Float32MultiArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import pyqtgraph as pg
from collections import deque


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

        self.body_state_subscriber = self.create_subscription(
            Float32MultiArray,
            'body_state',
            self.body_state_callback,
            10
        )

        # Publishers
        self.movement_publisher = self.create_publisher(Int32, 'movement_command', 10)

        # Service clients
        self.restart_client = self.create_client(Trigger, 'restart_simulation')

        # CvBridge for converting images
        self.bridge = CvBridge()

        # Store latest image
        self.latest_image = None

        # Store latest body state [x, y, z, roll, pitch, yaw]
        self.latest_body_state = None

        self.get_logger().info('GUI ROS Node initialized')

    def camera_callback(self, msg):
        """Handle incoming camera images."""
        try:
            # Convert ROS Image to numpy array (RGB)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def body_state_callback(self, msg):
        """Handle incoming body state data [x, y, z, roll, pitch, yaw]."""
        try:
            self.latest_body_state = list(msg.data)
        except Exception as e:
            self.get_logger().error(f'Failed to process body state: {e}')

    def publish_movement_command(self, command):
        """Publish movement command (0=no movement, 1=up, 2=down)."""
        msg = Int32()
        msg.data = command
        self.movement_publisher.publish(msg)

    def call_restart_service(self):
        """Call the restart simulation service."""
        if not self.restart_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Restart service not available')
            return

        request = Trigger.Request()
        future = self.restart_client.call_async(request)
        future.add_done_callback(self.restart_response_callback)

    def restart_response_callback(self, future):
        """Handle response from restart service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Restart successful: {response.message}')
            else:
                self.get_logger().error(f'Restart failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


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
        self.joystick_timer.start(10)  # Poll every 50ms

        # Setup timer for ROS2 spin
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.process_ros)
        self.ros_timer.start(10)  # Process ROS2 callbacks every 10ms

        # Setup timer for camera display update
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_display)
        self.camera_timer.start(100)  # Update display every 100ms (0.1 seconds)

        # Setup timer for body state display update
        self.body_state_timer = QTimer()
        self.body_state_timer.timeout.connect(self.update_body_state_display)
        self.body_state_timer.start(100)  # Update display every 100ms

        # Track current joystick state
        self.joystick_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False
        }

        # Track button states for edge detection (will be populated dynamically)
        self.button_states = {}

        # Track last published movement command
        self.last_movement_command = 0

        # Initialize XY position buffer (size 50) and plot
        self.xy_buffer_size = 50
        self.x_buffer = deque(maxlen=self.xy_buffer_size)
        self.y_buffer = deque(maxlen=self.xy_buffer_size)
        self.setup_xy_plot()

    def setup_ui(self):
        """Initialize UI elements."""
        # Hide operation tab initially
        self.tabWidget.setTabEnabled(self.tabWidget.indexOf(self.tab_operation), False)
        self.tabWidget.setCurrentWidget(self.tab_login)

        # Hide owner panel initially
        self.owner_panel.setVisible(False)

        # Set default brown images for arrow labels
        self.set_arrow_images()

    def setup_xy_plot(self):
        """Setup the XY position plot widget."""
        # Create plot widget
        self.xy_plot_widget = pg.PlotWidget()
        self.xy_plot_widget.setBackground('w')
        self.xy_plot_widget.setTitle("Robot XY Position", color='k', size='12pt')
        self.xy_plot_widget.setLabel('left', 'Y Position (m)', color='k')
        self.xy_plot_widget.setLabel('bottom', 'X Position (m)', color='k')
        self.xy_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.xy_plot_widget.setMinimumSize(300, 300)

        # Create the plot line - using scatter plot instead to avoid line artifacts
        self.xy_plot_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color='b', width=1),
            brush=pg.mkBrush(0, 100, 255, 200)
        )
        self.xy_plot_widget.addItem(self.xy_plot_scatter)

        # Also create a line plot for the trail
        self.xy_plot_line = self.xy_plot_widget.plot(
            pen=pg.mkPen(color='b', width=1.5, style=pg.QtCore.Qt.SolidLine),
            symbol=None
        )

        # Add plot widget to horizontalLayout_9 (next to 'Cuerpo' group)
        self.horizontalLayout_9.addWidget(self.xy_plot_widget)

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

    def update_body_state_display(self):
        """Update body state labels with latest data from ROS2."""
        if self.ros_node.latest_body_state is not None:
            try:
                # Extract data: [x, y, z, roll, pitch, yaw]
                x, y, z, roll, pitch, yaw = self.ros_node.latest_body_state

                # Update position labels (in meters)
                self.body_x_label.setText(f"{x:.4f}")
                self.body_y_label.setText(f"{y:.4f}")
                self.body_z_label.setText(f"{z:.4f}")

                # Update orientation labels (convert radians to degrees for display)
                self.body_roll_label.setText(f"{np.degrees(roll):.2f}°")
                self.body_pitch_label.setText(f"{np.degrees(pitch):.2f}°")
                self.body_yaw_label.setText(f"{np.degrees(yaw):.2f}°")

                # Update XY plot buffer and redraw
                self.x_buffer.append(x)
                self.y_buffer.append(y)

                # Update plot with buffered data
                if len(self.x_buffer) > 0:
                    x_data = list(self.x_buffer)
                    y_data = list(self.y_buffer)

                    # Update the line trail
                    self.xy_plot_line.setData(x_data, y_data)

                    # Update scatter plot points
                    self.xy_plot_scatter.setData(x_data, y_data)
            except Exception as e:
                print(f"Error updating body state display: {e}")

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

        # Check all buttons and print when any button is pressed (for debugging)
        num_buttons = self.joystick.get_numbuttons()
        for i in range(num_buttons):
            button_state = self.joystick.get_button(i)
            if button_state == 1:
                # Store button state if not tracked yet
                if i not in self.button_states:
                    self.button_states[i] = False

                # Detect rising edge (button press, not hold)
                if button_state and not self.button_states.get(i, False):
                    print(f"Button {i} pressed")

                    # Check if this is the restart button (button 0 for Android Gamepad)
                    if i == 0:
                        print("Restart button detected - requesting simulation restart")
                        self.ros_node.call_restart_service()

                # Update button state
                self.button_states[i] = button_state
            else:
                # Button released
                if i in self.button_states:
                    self.button_states[i] = False

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
        self.body_state_timer.stop()
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
