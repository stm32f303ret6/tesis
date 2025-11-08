#!/usr/bin/env python3
"""Joystick input handler for quadruped robot control."""

from dataclasses import dataclass
from typing import Optional

import pygame


@dataclass
class VelocityCommand:
    """Velocity command for robot locomotion."""

    vx: float = 0.0  # Forward/backward velocity (m/s)
    vy: float = 0.0  # Lateral velocity (m/s)
    omega: float = 0.0  # Yaw rotation velocity (rad/s)

    def is_stationary(self, threshold: float = 1e-4) -> bool:
        """Check if the command represents stationary state."""
        return abs(self.vx) < threshold and abs(self.vy) < threshold and abs(self.omega) < threshold


class JoystickInput:
    """Handles joystick input and converts to velocity commands."""

    def __init__(
        self,
        max_linear_vel: float = 0.15,
        max_angular_vel: float = 5.0,
        deadzone: float = 0.1,
        use_keyboard_fallback: bool = True,
    ) -> None:
        """
        Initialize joystick input handler.

        Args:
            max_linear_vel: Maximum linear velocity in m/s
            max_angular_vel: Maximum angular velocity in rad/s
            deadzone: Joystick deadzone threshold (0-1)
            use_keyboard_fallback: Enable keyboard control if no joystick
        """
        pygame.init()
        pygame.joystick.init()

        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.deadzone = deadzone
        self.use_keyboard_fallback = use_keyboard_fallback

        self.joystick: Optional[pygame.joystick.Joystick] = None
        self._init_joystick()

        # Keyboard state for fallback
        self.keyboard_vel = VelocityCommand()

    def _init_joystick(self) -> None:
        """Initialize the first available joystick."""
        joystick_count = pygame.joystick.get_count()

        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[JOYSTICK] Initialized: {self.joystick.get_name()}")
            print(f"[JOYSTICK] Axes: {self.joystick.get_numaxes()}")
        else:
            print("[JOYSTICK] No joystick detected")
            if self.use_keyboard_fallback:
                print("[JOYSTICK] Using keyboard fallback (WASD + QE for rotation)")

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick axis value."""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale to account for deadzone
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _read_joystick(self) -> VelocityCommand:
        """Read velocity command from joystick."""
        if self.joystick is None:
            return VelocityCommand()

        # Read axes (typical gamepad layout)
        # Axis 0: Left stick X (lateral)
        # Axis 1: Left stick Y (forward/backward)
        # Axis 2: Right stick X (rotation)
        # Note: Y-axis is typically inverted on gamepads

        lateral_raw = self.joystick.get_axis(0) if self.joystick.get_numaxes() > 0 else 0.0
        forward_raw = -self.joystick.get_axis(1) if self.joystick.get_numaxes() > 1 else 0.0
        rotation_raw = self.joystick.get_axis(2) if self.joystick.get_numaxes() > 2 else 0.0

        # Apply deadzone
        lateral = self._apply_deadzone(lateral_raw)
        forward = self._apply_deadzone(forward_raw)
        rotation = self._apply_deadzone(rotation_raw)

        # Scale to velocity limits
        vx = forward * self.max_linear_vel
        vy = lateral * self.max_linear_vel
        omega = rotation * self.max_angular_vel

        return VelocityCommand(vx=vx, vy=vy, omega=omega)

    def _read_keyboard(self) -> VelocityCommand:
        """Read velocity command from keyboard (fallback mode)."""
        keys = pygame.key.get_pressed()

        # WASD for linear movement
        vx = 0.0
        vy = 0.0
        if keys[pygame.K_w]:
            vx += self.max_linear_vel
        if keys[pygame.K_s]:
            vx -= self.max_linear_vel
        if keys[pygame.K_a]:
            vy -= self.max_linear_vel
        if keys[pygame.K_d]:
            vy += self.max_linear_vel

        # QE for rotation
        omega = 0.0
        if keys[pygame.K_q]:
            omega += self.max_angular_vel
        if keys[pygame.K_e]:
            omega -= self.max_angular_vel

        return VelocityCommand(vx=vx, vy=vy, omega=omega)

    def get_velocity_command(self) -> VelocityCommand:
        """
        Read and return current velocity command.

        Returns:
            VelocityCommand with current desired velocities
        """
        # Process pygame events
        pygame.event.pump()

        # Use joystick if available, otherwise keyboard
        if self.joystick is not None:
            return self._read_joystick()
        elif self.use_keyboard_fallback:
            return self._read_keyboard()
        else:
            return VelocityCommand()

    def shutdown(self) -> None:
        """Cleanup joystick resources."""
        if self.joystick is not None:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


# Example usage and testing
if __name__ == "__main__":
    import time

    joystick = JoystickInput(max_linear_vel=0.15, max_angular_vel=1.0)

    print("\n=== Joystick Input Test ===")
    print("Move joystick or use keyboard (WASD + QE)")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            cmd = joystick.get_velocity_command()

            if not cmd.is_stationary():
                print(
                    f"vx: {cmd.vx:6.3f} m/s, vy: {cmd.vy:6.3f} m/s, omega: {cmd.omega:6.3f} rad/s"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        joystick.shutdown()
