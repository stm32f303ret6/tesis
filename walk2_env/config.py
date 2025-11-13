"""Configuration for Walk2 environment."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Default configuration for Walk2 environments.

    Returns:
        ConfigDict with environment parameters
    """
    config = ml_collections.ConfigDict()

    # Timing
    config.control_dt = 0.02  # 50Hz control loop
    config.sim_dt = 0.002     # 500Hz physics simulation
    # n_substeps = control_dt / sim_dt = 10 substeps per control step

    # Action space
    config.action_scale = 0.5  # Scale actions from [-1,1] to joint range (radians)

    # PD control gains (tune based on robot dynamics)
    config.kp = 100.0  # Proportional gain
    config.kd = 2.0    # Derivative gain

    # Observation noise (for domain randomization)
    config.obs_noise = ml_collections.ConfigDict({
        'gyro': 0.01,          # Rad/s
        'accelerometer': 0.05,  # m/s^2
        'joint_angles': 0.01,  # Radians
        'joint_vels': 0.1,     # Rad/s
    })

    # Velocity command ranges for joystick task
    config.command_ranges = ml_collections.ConfigDict({
        'vx': (0.0, 1.0),      # Forward velocity (m/s)
        'vy': (-0.3, 0.3),     # Lateral velocity (m/s)
        'wz': (-0.5, 0.5),     # Angular velocity (rad/s)
    })

    # Reward weights
    config.reward_scales = ml_collections.ConfigDict({
        # Velocity tracking
        'tracking_vx': 1.0,
        'tracking_vy': 0.8,
        'tracking_wz': 0.5,

        # Stability
        'z_velocity': 0.2,     # Penalize vertical movement
        'orientation': 0.3,    # Penalize roll/pitch
        'height': 0.1,         # Maintain target height

        # Energy efficiency
        'torque': 0.001,       # Penalize high torques
        'power': 0.0005,       # Penalize power consumption
        'action_rate': 0.01,   # Encourage smooth actions

        # Gait quality
        'foot_slip': 0.05,     # Penalize foot slipping
        'foot_clearance': 0.02,  # Reward foot lift during swing

        # Safety
        'joint_limits': 0.1,   # Penalize joint limit violations
        'termination': 1.0,    # Large penalty for falling
    })

    # Target body height (meters)
    config.target_height = 0.08

    # Termination conditions
    config.termination = ml_collections.ConfigDict({
        'height_threshold': 0.04,  # Min height before episode ends (m)
        'roll_threshold': 0.8,      # Max roll angle (rad, ~45deg)
        'pitch_threshold': 0.8,     # Max pitch angle (rad, ~45deg)
    })

    # Domain randomization
    config.randomization = ml_collections.ConfigDict({
        'init_qpos_noise': 0.05,    # Radians
        'init_qvel_noise': 0.1,     # Rad/s
        'init_xy_range': 0.5,       # Meters (spawn position)
        'init_yaw_range': 3.14,     # Radians (initial heading)
    })

    return config
