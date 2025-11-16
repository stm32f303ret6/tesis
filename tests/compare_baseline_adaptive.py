#!/usr/bin/env python3
"""
Compare baseline gait controller vs adaptive policy performance.

This script runs two simulations:
1. Baseline: Pure gait controller without residuals (--baseline)
2. Adaptive: Trained policy with residual actions

Both simulations are run for the same duration, and trajectory data
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
    deterministic: bool = True
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

    Returns:
        Dictionary containing trajectory data
    """
    cmd = [
        "python3",
        "play_adaptive_policy.py",
        "--seconds", str(duration),
        "--save-trajectory", output_file,
    ]

    if baseline:
        cmd.append("--baseline")
        mode_name = "BASELINE"
    else:
        if model_path is None or normalize_path is None:
            raise ValueError("model_path and normalize_path required for adaptive mode")
        cmd.extend([
            "--model", model_path,
            "--normalize", normalize_path,
        ])
        mode_name = "ADAPTIVE"

    if deterministic:
        cmd.append("--deterministic")

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


def plot_comparison(baseline_data: Dict, adaptive_data: Dict, output_file: str = None):
    """
    Create side-by-side plots comparing baseline and adaptive trajectories.

    Args:
        baseline_data: Trajectory data from baseline simulation
        adaptive_data: Trajectory data from adaptive simulation
        output_file: Path to save the plot image
    """
    # Extract data
    baseline_times = [p["time"] for p in baseline_data["trajectory"]]
    baseline_x = [p["x"] for p in baseline_data["trajectory"]]

    adaptive_times = [p["time"] for p in adaptive_data["trajectory"]]
    adaptive_x = [p["x"] for p in adaptive_data["trajectory"]]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot baseline
    ax1.plot(baseline_times, baseline_x, 'b-', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('X Position (m)', fontsize=12)
    ax1.set_title('Baseline (Pure Gait Controller)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add start/end markers
    ax1.plot(baseline_times[0], baseline_x[0], 'go', markersize=10, label='Start')
    ax1.plot(baseline_times[-1], baseline_x[-1], 'ro', markersize=10, label='End')
    ax1.legend(loc='best')

    # Add stats
    baseline_distance = baseline_x[-1] - baseline_x[0]
    baseline_avg_vel = baseline_distance / baseline_data["duration"]
    ax1.text(0.05, 0.95,
             f'Distance: {baseline_distance:.3f}m\nAvg Vel: {baseline_avg_vel:.3f}m/s',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot adaptive
    ax2.plot(adaptive_times, adaptive_x, 'g-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('X Position (m)', fontsize=12)
    ax2.set_title('Adaptive (Policy with Residuals)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add start/end markers
    ax2.plot(adaptive_times[0], adaptive_x[0], 'go', markersize=10, label='Start')
    ax2.plot(adaptive_times[-1], adaptive_x[-1], 'ro', markersize=10, label='End')
    ax2.legend(loc='best')

    # Add stats
    adaptive_distance = adaptive_x[-1] - adaptive_x[0]
    adaptive_avg_vel = adaptive_distance / adaptive_data["duration"]
    ax2.text(0.05, 0.95,
             f'Distance: {adaptive_distance:.3f}m\nAvg Vel: {adaptive_avg_vel:.3f}m/s',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Match Y-axis limits for fair comparison
    all_x = baseline_x + adaptive_x
    y_min, y_max = min(all_x), max(all_x)
    y_margin = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)

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


def print_summary(baseline_data: Dict, adaptive_data: Dict):
    """Print comparison summary statistics."""
    baseline_traj = baseline_data["trajectory"]
    adaptive_traj = adaptive_data["trajectory"]

    baseline_distance = baseline_traj[-1]["x"] - baseline_traj[0]["x"]
    adaptive_distance = adaptive_traj[-1]["x"] - adaptive_traj[0]["x"]

    baseline_vel = baseline_distance / baseline_data["duration"]
    adaptive_vel = adaptive_distance / adaptive_data["duration"]

    improvement = ((adaptive_distance - baseline_distance) / abs(baseline_distance)) * 100

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Baseline':<15} {'Adaptive':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Duration (s)':<25} {baseline_data['duration']:<15.2f} {adaptive_data['duration']:<15.2f}")
    print(f"{'Data points':<25} {len(baseline_traj):<15} {len(adaptive_traj):<15}")
    print(f"{'Start X (m)':<25} {baseline_traj[0]['x']:<15.3f} {adaptive_traj[0]['x']:<15.3f}")
    print(f"{'End X (m)':<25} {baseline_traj[-1]['x']:<15.3f} {adaptive_traj[-1]['x']:<15.3f}")
    print(f"{'Distance traveled (m)':<25} {baseline_distance:<15.3f} {adaptive_distance:<15.3f} {adaptive_distance - baseline_distance:<+15.3f}")
    print(f"{'Average velocity (m/s)':<25} {baseline_vel:<15.3f} {adaptive_vel:<15.3f} {adaptive_vel - baseline_vel:<+15.3f}")
    print()
    print(f"Performance improvement: {improvement:+.1f}%")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline gait controller vs adaptive policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    baseline_file = "tests/trajectory_baseline.json"
    adaptive_file = "tests/trajectory_adaptive.json"

    try:
        # Run baseline simulation
        baseline_data = run_simulation(
            baseline=True,
            duration=args.seconds,
            output_file=baseline_file,
            deterministic=args.deterministic
        )

        # Run adaptive simulation
        adaptive_data = run_simulation(
            baseline=False,
            model_path=args.model,
            normalize_path=args.normalize,
            duration=args.seconds,
            output_file=adaptive_file,
            deterministic=args.deterministic
        )

        # Print summary
        print_summary(baseline_data, adaptive_data)

        # Create comparison plot
        print("Generating comparison plots...")
        plot_comparison(baseline_data, adaptive_data, output_file=args.output)

        print("\nComparison complete!")

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
