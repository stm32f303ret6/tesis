"""Recommended expanded parameter ranges based on training results.

Your current training showed the policy hitting the limits:
- step_height reached 0.06m maximum (hit the ceiling)
- cycle_time reached 0.6s minimum (hit the floor)

This suggests the policy wants to explore beyond current limits.
"""

# Copy this into AdaptiveGaitController.DEFAULT_RANGES
EXPANDED_RANGES = {
    # step_height: Currently hits 0.06m max frequently
    # Recommendation: Expand to 0.08m (still safe for IK)
    "step_height": (0.010, 0.080, 0.040),

    # step_length: Uses 0.03-0.072m range
    # Recommendation: Expand slightly to allow longer strides
    "step_length": (0.025, 0.100, 0.060),

    # cycle_time: Currently hits 0.6s min frequently
    # Recommendation: Allow faster gaits (0.4s) and slower (1.5s)
    "cycle_time": (0.400, 1.500, 0.800),

    # body_height: Barely adapts (0.04-0.045m)
    # Recommendation: Keep current range, it's fine
    "body_height": (0.035, 0.090, 0.050),
}

# If you want faster adaptation, increase delta scales
# Copy this into AdaptiveGaitEnv.PARAM_DELTA_SCALES
FASTER_DELTA_SCALES = {
    "step_height": 0.008,   # Up from 0.005 (60% faster)
    "step_length": 0.008,   # Up from 0.005 (60% faster)
    "cycle_time": 0.080,    # Up from 0.05 (60% faster)
    "body_height": 0.005,   # Up from 0.003 (66% faster)
}

# For very aggressive adaptation
AGGRESSIVE_DELTA_SCALES = {
    "step_height": 0.015,   # 3x faster
    "step_length": 0.015,   # 3x faster
    "cycle_time": 0.150,    # 3x faster
    "body_height": 0.010,   # 3x faster
}
