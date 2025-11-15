"""Plot learning curve from training run."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def plot_learning_curve(run_dir, window=50):
    """Plot learning curve with rolling average."""
    run_path = Path(run_dir)

    # Read all monitor files
    monitor_files = sorted(glob.glob(str(run_path / "monitor_*.csv.monitor.csv")))

    all_episodes = []
    for mfile in monitor_files:
        try:
            df = pd.read_csv(mfile, skiprows=1)
            all_episodes.append(df)
        except Exception as e:
            print(f"Error reading {mfile}: {e}")

    if not all_episodes:
        print("No monitor data found")
        return

    combined = pd.concat(all_episodes, ignore_index=True)
    combined = combined.sort_values('t').reset_index(drop=True)

    # Calculate rolling average
    combined['rolling_mean'] = combined['r'].rolling(window=window, min_periods=1).mean()
    combined['episode_num'] = range(len(combined))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Full learning curve
    ax1.plot(combined['episode_num'], combined['r'], alpha=0.3, label='Episode Return', linewidth=0.5)
    ax1.plot(combined['episode_num'], combined['rolling_mean'], 'r-', label=f'Rolling Mean ({window} episodes)', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Return')
    ax1.set_title(f'Learning Curve: {run_path.name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Last 25% of training (zoomed in)
    last_quarter_idx = len(combined) * 3 // 4
    last_quarter = combined.iloc[last_quarter_idx:]

    ax2.plot(last_quarter['episode_num'], last_quarter['r'], alpha=0.3, label='Episode Return', linewidth=0.5)
    ax2.plot(last_quarter['episode_num'], last_quarter['rolling_mean'], 'r-', label=f'Rolling Mean ({window} episodes)', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Return')
    ax2.set_title('Last 25% of Training (Zoomed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'learning_curve.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")

    # Analysis of final performance
    print(f"\n{'='*60}")
    print(f"Convergence Analysis")
    print(f"{'='*60}")

    # Last 5%, 10%, 20% comparison
    n = len(combined)
    last_5pct = combined.iloc[-n//20:]
    last_10pct = combined.iloc[-n//10:]
    last_20pct = combined.iloc[-n//5:]

    print(f"Last 20% episodes: Mean = {last_20pct['r'].mean():>10.2f}, Std = {last_20pct['r'].std():>10.2f}")
    print(f"Last 10% episodes: Mean = {last_10pct['r'].mean():>10.2f}, Std = {last_10pct['r'].std():>10.2f}")
    print(f"Last  5% episodes: Mean = {last_5pct['r'].mean():>10.2f}, Std = {last_5pct['r'].std():>10.2f}")

    # Check if still improving
    improvement_5_to_10 = last_5pct['r'].mean() - last_10pct['r'].mean()
    improvement_10_to_20 = last_10pct['r'].mean() - last_20pct['r'].mean()

    print(f"\nRecent improvement trends:")
    print(f"  10% to 5%:  {improvement_5_to_10:>10.2f}")
    print(f"  20% to 10%: {improvement_10_to_20:>10.2f}")

    if improvement_5_to_10 > 500 and improvement_10_to_20 > 500:
        print("\n✓ Still showing consistent improvement - consider training longer")
    elif improvement_5_to_10 > 500:
        print("\n~ Slight recent improvement - may benefit from more training")
    else:
        print("\n✓ Performance appears to have plateaued")

    return combined

if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/adaptive_gait_20251113_200227"
    plot_learning_curve(run_dir)
