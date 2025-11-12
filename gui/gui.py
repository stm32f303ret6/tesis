#!/usr/bin/env python3
# PyQt5: pip install PyQt5
# Pygame: pip install pygame

import sys
import os
import sqlite3
import pygame
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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

        # Track current joystick state
        self.joystick_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False
        }

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

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self.joystick_timer.stop()
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
