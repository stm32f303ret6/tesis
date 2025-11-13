"""Constants for Walk2 3DOF parallel SCARA quadruped environment."""

import pathlib

# Root paths
_WALK2_ROOT = pathlib.Path(__file__).parent.parent
_MODEL_ROOT = _WALK2_ROOT / "model"

# XML file paths
FLAT_TERRAIN_XML = str(_MODEL_ROOT / "world.xml")
ROUGH_TERRAIN_XML = str(_MODEL_ROOT / "world_train.xml")
ROBOT_XML = str(_MODEL_ROOT / "robot.xml")

# Robot structure - leg naming follows FL, FR, RL, RR convention
# (Front-Left, Front-Right, Rear-Left, Rear-Right)
LEGS = ['FL', 'RL', 'FR', 'RR']  # Order matches actuator indices in XML
BODY = 'robot'  # Main body link name

# Joint structure: each leg has 3 DOF
# - tilt: rotation around leg's X-axis for lateral positioning
# - shoulder_L: left arm of parallel SCARA
# - shoulder_R: right arm of parallel SCARA
JOINTS_PER_LEG = 3
NUM_ACTUATORS = len(LEGS) * JOINTS_PER_LEG  # 12 total

# Physical parameters (from CLAUDE.md and ik.py)
L1 = 0.045  # Upper link length (m)
L2 = 0.06   # Lower link length (m)
BASE_DIST = 0.021  # Distance between parallel arms (m)
MAX_REACH = L1 + L2  # 0.105m

# Default home position (approximate standing configuration)
# These are joint angles [tilt, shoulder_L, shoulder_R] for each leg
# in order: FL, RL, FR, RR
HOME_QPOS = [
    0.0, 1.0, 1.0,  # FL: level tilt, shoulders at ~60deg
    0.0, 1.0, 1.0,  # RL: level tilt, shoulders at ~60deg
    0.0, 1.0, 1.0,  # FR: level tilt, shoulders at ~60deg
    0.0, 1.0, 1.0,  # RR: level tilt, shoulders at ~60deg
]

# Sensor names (if defined in XML)
SENSORS = {
    'imu_gyro': 'imu_gyro',
    'imu_accel': 'imu_accelerometer',
    'body_linvel': 'body_linvel',
    'body_angvel': 'body_angvel',
}


def get_xml_path(terrain: str = 'rough') -> str:
    """Get XML path for specified terrain type.

    Args:
        terrain: 'flat' or 'rough'

    Returns:
        Path to XML file
    """
    if terrain == 'flat':
        return FLAT_TERRAIN_XML
    elif terrain == 'rough':
        return ROUGH_TERRAIN_XML
    else:
        raise ValueError(f"Unknown terrain type: {terrain}. Use 'flat' or 'rough'.")
