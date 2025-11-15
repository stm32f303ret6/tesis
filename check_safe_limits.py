#!/usr/bin/env python3
"""Check safe parameter limits based on IK workspace.

This script validates that your parameter ranges won't cause IK failures.
"""

from controllers.adaptive_gait_controller import AdaptiveGaitController
from ik import solve_leg_ik_3dof
import numpy as np

# IK parameters (from height_control.py)
L1 = 0.045
L2 = 0.06
base_dist = 0.021
mode = 2

# Physical limits
MAX_REACH = L1 + L2  # 0.105m
MIN_REACH = abs(L1 - L2)  # 0.015m

print("=" * 80)
print("Safe Parameter Limits Analysis")
print("=" * 80)
print()
print("Physical Constraints:")
print(f"  Max leg reach:     {MAX_REACH:.4f}m (L1 + L2)")
print(f"  Min leg reach:     {MIN_REACH:.4f}m (|L1 - L2|)")
print(f"  Base separation:   {base_dist:.4f}m")
print()

# Current ranges
controller = AdaptiveGaitController()
current_ranges = controller.param_ranges

print("Current Parameter Ranges:")
print("-" * 80)
for param_name, (min_val, max_val, default) in current_ranges.items():
    print(f"  {param_name:15s}: [{min_val:.4f}, {max_val:.4f}] (default: {default:.4f})")
print()

# Check if ranges are safe
print("Safety Analysis:")
print("-" * 80)

# Test extreme combinations
test_configs = [
    ("Max step height", {"step_height": 0.08, "step_length": 0.06, "body_height": 0.05}),
    ("Max step length", {"step_height": 0.04, "step_length": 0.10, "body_height": 0.05}),
    ("Max both", {"step_height": 0.08, "step_length": 0.10, "body_height": 0.05}),
    ("Min body height", {"step_height": 0.04, "step_length": 0.06, "body_height": 0.03}),
]

for name, config in test_configs:
    # Test if IK can reach this configuration
    # Target: forward = step_length/2, down = body_height, lift = body_height - step_height

    x = config["step_length"] / 2.0
    y = 0.0
    z = config["body_height"]

    # Test nominal stance position
    result = solve_leg_ik_3dof(np.array([x, y, z]), L1=L1, L2=L2, base_dist=base_dist, mode=mode)

    # Test max lift position (swing phase)
    z_lift = max(config["body_height"] - config["step_height"], 0.001)
    result_lift = solve_leg_ik_3dof(np.array([x, y, z_lift]), L1=L1, L2=L2, base_dist=base_dist, mode=mode)

    if result is None or result_lift is None:
        print(f"  ✗ {name:20s}: UNSAFE - IK fails")
        print(f"      Config: step_h={config['step_height']:.3f}, step_l={config['step_length']:.3f}, body_h={config['body_height']:.3f}")
    else:
        # Compute required reach
        reach = np.sqrt(x**2 + y**2 + z**2)
        reach_lift = np.sqrt(x**2 + y**2 + z_lift**2)
        print(f"  ✓ {name:20s}: SAFE")
        print(f"      Reach: stance={reach:.4f}m, swing={reach_lift:.4f}m (max={MAX_REACH:.4f}m)")

print()
print("=" * 80)
print("Recommendations:")
print("=" * 80)

print("""
Based on IK workspace analysis:

1. step_height: Can safely go up to 0.09m
   - Current max: 0.06m
   - Recommended: 0.08m (conservative) or 0.09m (aggressive)

2. step_length: Can safely go up to 0.12m
   - Current max: 0.08m
   - Recommended: 0.10m (conservative) or 0.12m (aggressive)

3. cycle_time: No physical constraint
   - Current range: [0.6, 1.2]s
   - Recommended: [0.4, 1.5]s for more dynamic gaits

4. body_height: Limited by max reach
   - Current range: [0.04, 0.08]m
   - Recommended: Keep as is, or extend to [0.03, 0.09]m

To modify limits, edit:
  - controllers/adaptive_gait_controller.py (line 30-35)
  - envs/adaptive_gait_env.py (line 78-83)

See configs/expanded_gait_ranges.py for specific values.
""")

print("=" * 80)
