#!/usr/bin/env python3
"""
Compare baseline gait controller vs adaptive policy performance across terrains.

This script runs three simulations and records time vs position X:
1. Step 1 (Left):   Baseline on flat terrain (world.xml)
2. Step 2 (Middle): Baseline on rough terrain (world_train.xml)
3. Step 3 (Right):  Adaptive/RL residual on rough terrain (world_train.xml)

All simulations are run for the same duration, and trajectory data
(timestamp and X position) are recorded and plotted side-by-side.

Usage:
    python3 tests/compare_baseline_adaptive.py \\
        --model runs/adaptive_gait_20251115_180640/final_model.zip \\
        --normalize runs/adaptive_gait_20251115_180640/vec_normalize.pkl \\
        --seconds 17
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def run_simulation(
    baseline: bool,
    model_path: str = None,
    normalize_path: str = None,
    duration: float = 17.0,
    output_file: str = None,
    deterministic: bool = True,
    flat_terrain: bool = False
) -> Dict:
    """
    Run a single simulation and save trajectory data.

    Args:
        baseline: If True, run baseline mode (zero actions)
        model_path: Path to trained model (.zip)
        normalize_path: Path to VecNormalize stats (.pkl)
        duration: Simulation duration in seconds
        output_file: Path to save trajectory JSON
        deterministic: Use deterministic policy actions
        flat_terrain: If True, use flat terrain (world.xml) instead of rough (world_train.xml)

    Returns:
        Dictionary containing trajectory data
    """
    cmd = [
        "python3",
        "play_adaptive_policy.py",
        "--seconds", str(duration),
        "--save-trajectory", output_file,
        "--fullscreen"
    ]

    if baseline:
        cmd.append("--baseline")
        terrain_str = "FLAT" if flat_terrain else "ROUGH"
        mode_name = f"BASELINE ({terrain_str})"
    else:
        if model_path is None or normalize_path is None:
            raise ValueError("model_path and normalize_path required for adaptive mode")
        cmd.extend([
            "--model", model_path,
            "--normalize", normalize_path,
        ])
        terrain_str = "FLAT" if flat_terrain else "ROUGH"
        mode_name = f"ADAPTIVE ({terrain_str})"

    if deterministic:
        cmd.append("--deterministic")

    if flat_terrain:
        cmd.append("--flat")

    print("=" * 80)
    print(f"Running {mode_name} simulation...")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run simulation
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"{mode_name} simulation failed with exit code {result.returncode}")

    # Load trajectory data
    with open(output_file, 'r') as f:
        data = json.load(f)

    print(f"\n{mode_name} simulation complete!")
    print(f"  Duration: {data['duration']:.2f}s")
    print(f"  Data points: {len(data['trajectory'])}")
    print()

    return data


def plot_comparison(baseline_flat_data: Dict, baseline_rough_data: Dict, adaptive_rough_data: Dict, output_file: str = None):
    """
    Create three-panel comparison plot.

    Args:
        baseline_flat_data: Step 1 - Baseline on flat terrain (world.xml)
        baseline_rough_data: Step 2 - Baseline on rough terrain (world_train.xml)
        adaptive_rough_data: Step 3 - Adaptive/RL on rough terrain (world_train.xml)
        output_file: Path to save the plot image
    """
    # Extract data for all three simulations
    baseline_flat_times = [p["time"] for p in baseline_flat_data["trajectory"]]
    baseline_flat_x = [p["x"] for p in baseline_flat_data["trajectory"]]

    baseline_rough_times = [p["time"] for p in baseline_rough_data["trajectory"]]
    baseline_rough_x = [p["x"] for p in baseline_rough_data["trajectory"]]

    adaptive_rough_times = [p["time"] for p in adaptive_rough_data["trajectory"]]
    adaptive_rough_x = [p["x"] for p in adaptive_rough_data["trajectory"]]

    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Baseline (Flat Terrain) - LEFT
    ax1.plot(baseline_flat_times, baseline_flat_x, 'b-', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('X Position (m)', fontsize=12)
    ax1.set_title('Step 1: Baseline\n(Pure Gait - Flat Terrain)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add start/end markers
    ax1.plot(baseline_flat_times[0], baseline_flat_x[0], 'go', markersize=10, label='Start')
    ax1.plot(baseline_flat_times[-1], baseline_flat_x[-1], 'ro', markersize=10, label='End')
    ax1.legend(loc='best', fontsize=9)

    # Add stats
    baseline_flat_distance = baseline_flat_x[-1] - baseline_flat_x[0]
    baseline_flat_avg_vel = baseline_flat_distance / baseline_flat_data["duration"]
    ax1.text(0.05, 0.95,
             f'Distance: {baseline_flat_distance:.3f}m\nAvg Vel: {baseline_flat_avg_vel:.3f}m/s',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Plot 2: Baseline (Rough Terrain) - MIDDLE
    ax2.plot(baseline_rough_times, baseline_rough_x, 'red', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('X Position (m)', fontsize=12)
    ax2.set_title('Step 2: Baseline\n(Pure Gait - Rough Terrain)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add start/end markers
    ax2.plot(baseline_rough_times[0], baseline_rough_x[0], 'go', markersize=10, label='Start')
    ax2.plot(baseline_rough_times[-1], baseline_rough_x[-1], 'ro', markersize=10, label='End')
    ax2.legend(loc='best', fontsize=9)

    # Add stats
    baseline_rough_distance = baseline_rough_x[-1] - baseline_rough_x[0]
    baseline_rough_avg_vel = baseline_rough_distance / baseline_rough_data["duration"]
    degradation = ((baseline_rough_distance - baseline_flat_distance) / abs(baseline_flat_distance)) * 100
    ax2.text(0.05, 0.95,
             f'Distance: {baseline_rough_distance:.3f}m\nAvg Vel: {baseline_rough_avg_vel:.3f}m/s\nvs Flat: {degradation:+.1f}%',
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 3: Adaptive (Rough Terrain) - RIGHT
    ax3.plot(adaptive_rough_times, adaptive_rough_x, 'g-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('X Position (m)', fontsize=12)
    ax3.set_title('Step 3: Adaptive RL\n(Policy + Residuals - Rough Terrain)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add start/end markers
    ax3.plot(adaptive_rough_times[0], adaptive_rough_x[0], 'go', markersize=10, label='Start')
    ax3.plot(adaptive_rough_times[-1], adaptive_rough_x[-1], 'ro', markersize=10, label='End')
    ax3.legend(loc='best', fontsize=9)

    # Add stats
    adaptive_rough_distance = adaptive_rough_x[-1] - adaptive_rough_x[0]
    adaptive_rough_avg_vel = adaptive_rough_distance / adaptive_rough_data["duration"]
    improvement_vs_baseline_rough = ((adaptive_rough_distance - baseline_rough_distance) / abs(baseline_rough_distance)) * 100
    ax3.text(0.05, 0.95,
             f'Distance: {adaptive_rough_distance:.3f}m\nAvg Vel: {adaptive_rough_avg_vel:.3f}m/s\nvs Step2: {improvement_vs_baseline_rough:+.1f}%',
             transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Match Y-axis limits for fair comparison
    all_x = baseline_flat_x + baseline_rough_x + adaptive_rough_x
    y_min, y_max = min(all_x), max(all_x)
    y_margin = (y_max - y_min) * 0.1
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = "tests/baseline_vs_adaptive_comparison.png"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also show the plot
    plt.show()


def print_summary(baseline_flat_data: Dict, baseline_rough_data: Dict, adaptive_rough_data: Dict):
    """Print comparison summary statistics for all three simulations."""
    baseline_flat_traj = baseline_flat_data["trajectory"]
    baseline_rough_traj = baseline_rough_data["trajectory"]
    adaptive_rough_traj = adaptive_rough_data["trajectory"]

    baseline_flat_distance = baseline_flat_traj[-1]["x"] - baseline_flat_traj[0]["x"]
    baseline_rough_distance = baseline_rough_traj[-1]["x"] - baseline_rough_traj[0]["x"]
    adaptive_rough_distance = adaptive_rough_traj[-1]["x"] - adaptive_rough_traj[0]["x"]

    baseline_flat_vel = baseline_flat_distance / baseline_flat_data["duration"]
    baseline_rough_vel = baseline_rough_distance / baseline_rough_data["duration"]
    adaptive_rough_vel = adaptive_rough_distance / adaptive_rough_data["duration"]

    degradation = ((baseline_rough_distance - baseline_flat_distance) / abs(baseline_flat_distance)) * 100
    improvement = ((adaptive_rough_distance - baseline_rough_distance) / abs(baseline_rough_distance)) * 100

    print("\n" + "=" * 95)
    print("COMPARISON SUMMARY - THREE SIMULATIONS")
    print("=" * 95)
    print()
    print(f"{'Metric':<30} {'Step 1: Baseline':<20} {'Step 2: Baseline':<20} {'Step 3: Adaptive':<20}")
    print(f"{'':30} {'(Flat)':<20} {'(Rough)':<20} {'(Rough)':<20}")
    print("-" * 95)
    print(f"{'Duration (s)':<30} {baseline_flat_data['duration']:<20.2f} {baseline_rough_data['duration']:<20.2f} {adaptive_rough_data['duration']:<20.2f}")
    print(f"{'Data points':<30} {len(baseline_flat_traj):<20} {len(baseline_rough_traj):<20} {len(adaptive_rough_traj):<20}")
    print(f"{'Start X (m)':<30} {baseline_flat_traj[0]['x']:<20.3f} {baseline_rough_traj[0]['x']:<20.3f} {adaptive_rough_traj[0]['x']:<20.3f}")
    print(f"{'End X (m)':<30} {baseline_flat_traj[-1]['x']:<20.3f} {baseline_rough_traj[-1]['x']:<20.3f} {adaptive_rough_traj[-1]['x']:<20.3f}")
    print(f"{'Distance traveled (m)':<30} {baseline_flat_distance:<20.3f} {baseline_rough_distance:<20.3f} {adaptive_rough_distance:<20.3f}")
    print(f"{'Average velocity (m/s)':<30} {baseline_flat_vel:<20.3f} {baseline_rough_vel:<20.3f} {adaptive_rough_vel:<20.3f}")
    print()
    print("Performance Comparison:")
    print(f"  Step 2 vs Step 1 (Rough vs Flat):      {degradation:+.1f}%")
    print(f"  Step 3 vs Step 2 (Adaptive vs Rough):  {improvement:+.1f}%")
    print("=" * 95)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline gait controller vs adaptive policy across terrains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs three simulations and creates a side-by-side comparison:
  Step 1 (Left):   Baseline on flat terrain (world.xml)
  Step 2 (Middle): Baseline on rough terrain (world_train.xml)
  Step 3 (Right):  Adaptive/RL residual on rough terrain (world_train.xml)

Example usage:
  python3 tests/compare_baseline_adaptive.py \\
      --model runs/adaptive_gait_20251115_180640/final_model.zip \\
      --normalize runs/adaptive_gait_20251115_180640/vec_normalize.pkl \\
      --seconds 17
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--normalize",
        type=str,
        required=True,
        help="Path to VecNormalize stats (.pkl)"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=17.0,
        help="Duration for each simulation (default: 17.0s)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions (default: True)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/baseline_vs_adaptive_comparison.png",
        help="Output path for comparison plot"
    )

    args = parser.parse_args()

    # Temporary files for trajectory data
    baseline_flat_file = "tests/trajectory_step1_baseline_flat.json"
    baseline_rough_file = "tests/trajectory_step2_baseline_rough.json"
    adaptive_rough_file = "tests/trajectory_step3_adaptive_rough.json"

    try:
        # Run Step 1: Baseline on flat terrain
        print("\n" + "#" * 80)
        print("# STEP 1: Baseline (Pure Gait Controller - Flat Terrain)")
        print("#" * 80 + "\n")
        baseline_flat_data = run_simulation(
            baseline=True,
            duration=args.seconds,
            output_file=baseline_flat_file,
            deterministic=args.deterministic,
            flat_terrain=True
        )

        # Run Step 2: Baseline on rough terrain
        print("\n" + "#" * 80)
        print("# STEP 2: Baseline (Pure Gait Controller - Rough Terrain)")
        print("#" * 80 + "\n")
        baseline_rough_data = run_simulation(
            baseline=True,
            duration=args.seconds,
            output_file=baseline_rough_file,
            deterministic=args.deterministic,
            flat_terrain=False
        )

        # Run Step 3: Adaptive on rough terrain
        print("\n" + "#" * 80)
        print("# STEP 3: Adaptive RL (Policy with Residuals - Rough Terrain)")
        print("#" * 80 + "\n")
        adaptive_rough_data = run_simulation(
            baseline=False,
            model_path=args.model,
            normalize_path=args.normalize,
            duration=args.seconds,
            output_file=adaptive_rough_file,
            deterministic=args.deterministic,
            flat_terrain=False
        )

        # Print summary
        print_summary(baseline_flat_data, baseline_rough_data, adaptive_rough_data)

        # Create comparison plot
        print("Generating three-panel comparison plot...")
        plot_comparison(baseline_flat_data, baseline_rough_data, adaptive_rough_data, output_file=args.output)

        print("\nComparison complete!")
        print(f"\nResults:")
        print(f"  - Step 1 trajectory: {baseline_flat_file}")
        print(f"  - Step 2 trajectory: {baseline_rough_file}")
        print(f"  - Step 3 trajectory: {adaptive_rough_file}")
        print(f"  - Comparison plot:   {args.output}")

    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
