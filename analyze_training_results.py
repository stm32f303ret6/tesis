"""Analyze training results from monitor files."""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_training_run(run_dir):
    """Analyze monitor files from a training run."""
    run_path = Path(run_dir)

    # Read all monitor files
    monitor_files = sorted(glob.glob(str(run_path / "monitor_*.csv.monitor.csv")))

    all_episodes = []
    for mfile in monitor_files:
        try:
            # Read monitor file, skipping the first line (metadata)
            df = pd.read_csv(mfile, skiprows=1)
            all_episodes.append(df)
        except Exception as e:
            print(f"Error reading {mfile}: {e}")

    # Combine all episodes
    if not all_episodes:
        print("No monitor data found")
        return

    combined = pd.concat(all_episodes, ignore_index=True)
    combined = combined.sort_values('t').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Training Run Analysis: {run_path.name}")
    print(f"{'='*60}")
    print(f"\nTotal episodes: {len(combined)}")
    print(f"Total environments: {len(monitor_files)}")

    # Overall statistics
    print(f"\n{'Overall Statistics':^60}")
    print(f"{'-'*60}")
    print(f"Mean episode return:     {combined['r'].mean():>15.2f}")
    print(f"Std episode return:      {combined['r'].std():>15.2f}")
    print(f"Min episode return:      {combined['r'].min():>15.2f}")
    print(f"Max episode return:      {combined['r'].max():>15.2f}")
    print(f"Mean episode length:     {combined['l'].mean():>15.2f}")

    # First 10% vs Last 10% comparison
    n_episodes = len(combined)
    first_10pct = combined.iloc[:n_episodes//10]
    last_10pct = combined.iloc[-n_episodes//10:]

    print(f"\n{'Learning Progress':^60}")
    print(f"{'-'*60}")
    print(f"First 10% episodes:")
    print(f"  Mean return:           {first_10pct['r'].mean():>15.2f}")
    print(f"  Best return:           {first_10pct['r'].max():>15.2f}")
    print(f"\nLast 10% episodes:")
    print(f"  Mean return:           {last_10pct['r'].mean():>15.2f}")
    print(f"  Best return:           {last_10pct['r'].max():>15.2f}")
    print(f"\nImprovement:")
    print(f"  Mean return change:    {last_10pct['r'].mean() - first_10pct['r'].mean():>15.2f}")
    print(f"  Percentage improvement:{(last_10pct['r'].mean() - first_10pct['r'].mean()) / abs(first_10pct['r'].mean()) * 100:>14.1f}%")

    # Quartile analysis
    print(f"\n{'Performance by Training Phase':^60}")
    print(f"{'-'*60}")
    q1 = combined.iloc[:n_episodes//4]
    q2 = combined.iloc[n_episodes//4:n_episodes//2]
    q3 = combined.iloc[n_episodes//2:3*n_episodes//4]
    q4 = combined.iloc[3*n_episodes//4:]

    for i, (name, quarter) in enumerate([("Q1 (0-25%)", q1), ("Q2 (25-50%)", q2),
                                          ("Q3 (50-75%)", q3), ("Q4 (75-100%)", q4)], 1):
        print(f"{name:<15} Mean: {quarter['r'].mean():>12.2f}  Best: {quarter['r'].max():>12.2f}")

    # Top 10 best episodes
    print(f"\n{'Top 10 Best Episodes':^60}")
    print(f"{'-'*60}")
    top_10 = combined.nlargest(10, 'r')[['r', 'l', 't']].reset_index(drop=True)
    for idx, row in top_10.iterrows():
        print(f"{idx+1:2d}. Return: {row['r']:>12.2f}  Length: {row['l']:>6.0f}  Time: {row['t']:>10.1f}s")

    # Training duration
    print(f"\n{'Training Duration':^60}")
    print(f"{'-'*60}")
    total_time = combined['t'].max() - combined['t'].min()
    print(f"Total training time:     {total_time:>15.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Episodes per minute:     {len(combined) / (total_time/60):>15.2f}")

    return combined

if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/adaptive_gait_20251113_200227"
    analyze_training_run(run_dir)
