"""Run the full pipeline: generate data -> features -> anomaly scores -> model."""

import subprocess
import sys


def run(script):
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script], cwd=".", capture_output=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    run("src/generate_data.py")
    run("src/feature_engineering.py")
    run("src/anomaly_detection.py")
    run("src/train_model.py")
