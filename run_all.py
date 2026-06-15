"""
run_all.py

Sequentially dispatches json_runner.py on every JSON job file in samples/.

This is a stand-in for what will eventually be a Slurm array job (one task
per JSON file). For now it just loops and calls json_runner.py one at a time.

Usage:
    python run_all.py
"""

import glob
import os
import subprocess
import sys

SAMPLES_DIR = "samples"


def main():
    job_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.json")))

    if not job_files:
        print(f"No JSON files found in '{SAMPLES_DIR}/'")
        return

    print(f"Found {len(job_files)} job files in '{SAMPLES_DIR}/'")

    for i, job_file in enumerate(job_files, start=1):
        print(f"\n[{i}/{len(job_files)}] Running {job_file}")
        result = subprocess.run([sys.executable, "Json_runner.py", job_file])
        if result.returncode != 0:
            print(f"  WARNING: {job_file} exited with code {result.returncode}")

    print("\nAll jobs dispatched.")


if __name__ == "__main__":
    main()