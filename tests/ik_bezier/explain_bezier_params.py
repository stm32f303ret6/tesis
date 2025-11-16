#!/usr/bin/env python3
"""
Visual explanation of bezier gait parameters.
Creates annotated diagrams showing what each parameter controls.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from test_bezier_gait import generate_gait_trajectory

# Create figure with multiple examples
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parámetros de la marcha Bézier explicados', fontsize=16, fontweight='bold')

# Default parameters
default_params = {
    'num_points': 100,
    'step_height': 0.03,
    'step_length': 0.04,
    'stance_height': -0.08
}

def plot_trajectory_with_annotations(ax, params, title, highlight=None):
    """Plot trajectory with parameter annotations."""
    traj = generate_gait_trajectory(**params)

    # Split into swing and stance
    mid = len(traj) // 2
    swing = traj[:mid]
    stance = traj[mid:]

    # Plot trajectory
    ax.plot(swing[:, 0], swing[:, 2], 'r-', linewidth=3, label='Fase de vuelo')
    ax.plot(stance[:, 0], stance[:, 2], 'g-', linewidth=3, label='Fase de apoyo')

    # Mark start and end
    ax.plot(traj[0, 0], traj[0, 2], 'go', markersize=12, label='Inicio')
    ax.plot(traj[mid-1, 0], traj[mid-1, 2], 'ro', markersize=12, label='Fin del vuelo')

    # Annotate parameters
    step_length = params['step_length']
    step_height = params['step_height']
    stance_height = params['stance_height']

    # Step length annotation
    if highlight is None or highlight == 'step_length':
        ax.annotate('', xy=(step_length/2, stance_height),
                   xytext=(-step_length/2, stance_height),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
        ax.text(0, stance_height - 0.008, f'longitud de paso\n{step_length*1000:.0f} mm',
                ha='center', va='top', fontsize=10, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Step height annotation
    if highlight is None or highlight == 'step_height':
        max_height = stance_height + step_height
        ax.annotate('', xy=(0, max_height), xytext=(0, stance_height),
                   arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        ax.text(0.005, stance_height + step_height/2,
                f'altura del paso\n{step_height*1000:.0f} mm',
                ha='left', va='center', fontsize=10, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Stance height annotation
    if highlight is None or highlight == 'stance_height':
        ax.annotate('', xy=(-step_length/2 - 0.01, 0),
                   xytext=(-step_length/2 - 0.01, stance_height),
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(-step_length/2 - 0.015, stance_height/2,
                f'altura de apoyo\n{stance_height*1000:.0f} mm\n(desde la base)',
                ha='right', va='center', fontsize=10, color='purple', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Draw base line (z=0)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Base (z = 0)')

    # Mark highest point
    max_idx = np.argmax(swing[:, 2])
    ax.plot(swing[max_idx, 0], swing[max_idx, 2], 'r*', markersize=15)

    ax.set_xlabel('Posición X (m)', fontsize=11)
    ax.set_ylabel('Posición Z (m)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('equal')
    ax.set_xlim(-0.06, 0.06)
    ax.set_ylim(-0.12, 0.02)

# 1. Default parameters with all annotations
plot_trajectory_with_annotations(axes[0, 0], default_params,
                                'Parámetros por defecto\n(todos los valores etiquetados)')

# 2. Varying step_height
params_high_step = default_params.copy()
params_high_step['step_height'] = 0.05  # 50mm instead of 30mm
plot_trajectory_with_annotations(axes[0, 1], params_high_step,
                                f'Altura de paso aumentada: {params_high_step["step_height"]*1000:.0f} mm\n(mayor despeje)',
                                highlight='step_height')

# 3. Varying step_length
params_long_step = default_params.copy()
params_long_step['step_length'] = 0.06  # 60mm instead of 40mm
plot_trajectory_with_annotations(axes[1, 0], params_long_step,
                                f'Longitud de paso aumentada: {params_long_step["step_length"]*1000:.0f} mm\n(paso más largo)',
                                highlight='step_length')

# 4. Varying stance_height
params_low_stance = default_params.copy()
params_low_stance['stance_height'] = -0.06  # -60mm instead of -80mm
plot_trajectory_with_annotations(axes[1, 1], params_low_stance,
                                f'Altura de apoyo modificada: {params_low_stance["stance_height"]*1000:.0f} mm\n(cuerpo más alto)',
                                highlight='stance_height')

plt.tight_layout()
plt.savefig('bezier_params_explained.png', dpi=150, bbox_inches='tight')
print("Saved explanation diagram to bezier_params_explained.png")

# Create a text summary
summary = """
=== BEZIER GAIT PARAMETERS EXPLANATION ===

The gait cycle has TWO phases:

1. SWING PHASE (50% of cycle) - Foot in the air
   - Uses a cubic Bezier curve with 4 control points
   - Lifts the foot, moves it forward, then sets it down

2. STANCE PHASE (50% of cycle) - Foot on the ground
   - Linear motion from front to back
   - Simulates the body moving forward over the planted foot

PARAMETERS:

1. num_points (default: 100)
   - Total number of trajectory points in one complete gait cycle
   - Higher = smoother trajectory but more computation
   - 50 points for swing, 50 points for stance

2. step_length (default: 0.04m = 40mm)
   - How far forward/backward the foot travels
   - Horizontal distance from back position to front position
   - Larger value = longer stride
   - Range at step_length/2 to -step_length/2

3. step_height (default: 0.03m = 30mm)
   - Maximum height the foot lifts during swing
   - Measured from stance_height
   - Larger value = higher foot clearance over obstacles
   - Peak occurs near middle of swing phase

4. stance_height (default: -0.08m = -80mm)
   - Height of foot when on ground (negative = below base)
   - This sets the robot's body height
   - More negative = robot squats lower
   - Less negative = robot stands taller
   - Must be within IK reachable range!

BEZIER CONTROL POINTS (automatically calculated):

  P0: [-step_length/2, 0, stance_height]           ← Start: back, on ground
  P1: [-step_length/3, 0, stance_height + step_height]  ← Lift back
  P2: [+step_length/3, 0, stance_height + step_height]  ← High forward
  P3: [+step_length/2, 0, stance_height]           ← End: front, on ground

The Bezier curve creates a smooth arc from P0 → P3 influenced by P1 and P2.

COORDINATE SYSTEM:
  X: Forward/backward (positive = forward)
  Y: Lateral/sideways (0 for basic gait)
  Z: Vertical (negative = down from base, 0 = base level)

TYPICAL VALUES FOR YOUR ROBOT:
  - Max reach: L1 + L2 = 0.045 + 0.06 = 0.105m (105mm)
  - Safe stance_height range: -0.04 to -0.09m (-40mm to -90mm)
  - Safe step_length range: 0.02 to 0.06m (20mm to 60mm)
  - Safe step_height range: 0.01 to 0.05m (10mm to 50mm)
"""

print(summary)

with open('bezier_params_explanation.txt', 'w') as f:
    f.write(summary)
print("\nSaved text explanation to bezier_params_explanation.txt")
