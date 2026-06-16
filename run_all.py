"""
run_all.py

Sequentially dispatches json_runner.py on every JSON job file in a samples directory.

This is a stand-in for what will eventually be a Slurm array job (one task
per JSON file). For now it just loops and calls json_runner.py one at a time.

Usage:
    python run_all.py                          # uses default 'samples/' dir
    python run_all.py --samples test_samples   # uses test_samples/ dir
"""

import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        default="samples",
        help="Directory containing JSON job files (default: samples)",
    )
    args = parser.parse_args()

    job_files = sorted(glob.glob(os.path.join(args.samples, "*.json")))

    if not job_files:
        print(f"No JSON files found in '{args.samples}/'")
        return

    print(f"Found {len(job_files)} job files in '{args.samples}/'")

    for i, job_file in enumerate(job_files, start=1):
        print(f"\n[{i}/{len(job_files)}] Running {job_file}")
        result = subprocess.run([sys.executable, "json_runner.py", job_file])
        if result.returncode != 0:
            print(f"  WARNING: {job_file} exited with code {result.returncode}")

    print("\nAll jobs dispatched.")


if __name__ == "__main__":
    main()
