"""Walk2 3DOF parallel SCARA quadruped environment for MuJoCo Playground."""

from walk2_env.base import Walk2Env
from walk2_env.joystick import Walk2Joystick
from walk2_env.config import get_config
from walk2_env import constants

__all__ = [
    'Walk2Env',
    'Walk2Joystick',
    'get_config',
    'constants',
    'load',
]

# Available environments
ALL_ENVS = (
    'Walk2JoystickFlat',
    'Walk2JoystickRough',
)

_ENVS = {
    'Walk2JoystickFlat': Walk2Joystick,
    'Walk2JoystickRough': Walk2Joystick,
}

_TERRAINS = {
    'Walk2JoystickFlat': 'flat',
    'Walk2JoystickRough': 'rough',
}


def load(env_name: str, config=None, **kwargs):
    """Load Walk2 environment.

    Args:
        env_name: Environment name (e.g., 'Walk2JoystickRough')
        config: Optional config override
        **kwargs: Additional arguments passed to environment

    Returns:
        Environment instance

    Example:
        >>> env = walk2_env.load('Walk2JoystickRough')
        >>> rng = jax.random.PRNGKey(0)
        >>> state = env.reset(rng)
    """
    if env_name not in ALL_ENVS:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available: {ALL_ENVS}"
        )

    # Get environment class
    cls = _ENVS[env_name]

    # Get default config
    cfg = get_config()

    # Apply config overrides
    if config is not None:
        cfg.update(config)

    # Get XML path for terrain type
    terrain = _TERRAINS[env_name]
    xml_path = constants.get_xml_path(terrain)

    # Create environment
    return cls(xml_path=xml_path, config=cfg, **kwargs)
