"""
Synthetic industrial sensor data generator.

Simulates a fleet of machines, each instrumented with vibration, temperature,
and current sensors. Each machine runs for a random number of cycles, and a
subset of machines degrade towards failure (rising vibration/temperature,
fluctuating current) while others run healthily for the full observation
window (censored / no failure).

This mimics the structure of real predictive-maintenance datasets (e.g. NASA
C-MAPSS) but is fully synthetic and reproducible, which keeps this project
self-contained and free of licensing issues.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_MACHINES = 60
MIN_CYCLES = 120
MAX_CYCLES = 260
FAILURE_FRACTION = 0.65  # fraction of machines that actually fail in-window


def generate_machine_series(machine_id: int, rng: np.random.Generator, will_fail: bool):
    n_cycles = rng.integers(MIN_CYCLES, MAX_CYCLES)

    # Baseline healthy operating levels, with per-machine variation
    base_vibration = rng.normal(2.0, 0.15)      # mm/s
    base_temperature = rng.normal(55.0, 2.0)    # deg C
    base_current = rng.normal(10.0, 0.5)        # Amps

    vibration = np.full(n_cycles, base_vibration)
    temperature = np.full(n_cycles, base_temperature)
    current = np.full(n_cycles, base_current)

    failure_cycle = None
    if will_fail:
        # Degradation begins at a random point in the machine's life
        degradation_start = rng.integers(int(n_cycles * 0.35), int(n_cycles * 0.75))
        failure_cycle = n_cycles - 1  # machine fails at the very last observed cycle

        degrade_len = failure_cycle - degradation_start
        t = np.arange(degrade_len + 1)

        # Nonlinear (accelerating) degradation trends
        vib_trend = 3.5 * (t / degrade_len) ** 2
        temp_trend = 18.0 * (t / degrade_len) ** 2
        current_trend = 4.0 * (t / degrade_len) ** 1.5

        vibration[degradation_start:] += vib_trend
        temperature[degradation_start:] += temp_trend
        # current becomes noisier/fluctuates more as bearings/motor wear
        current[degradation_start:] += current_trend + rng.normal(
            0, 0.4, size=degrade_len + 1
        )

    # Sensor noise
    vibration += rng.normal(0, 0.08, size=n_cycles)
    temperature += rng.normal(0, 0.6, size=n_cycles)
    current += rng.normal(0, 0.25, size=n_cycles)

    df = pd.DataFrame(
        {
            "machine_id": machine_id,
            "cycle": np.arange(1, n_cycles + 1),
            "vibration_mm_s": vibration,
            "temperature_c": temperature,
            "current_a": current,
        }
    )
    df["failed"] = 1 if will_fail else 0
    df["failure_cycle"] = failure_cycle if will_fail else np.nan
    return df


def main():
    rng = np.random.default_rng(RNG_SEED)
    n_failing = int(N_MACHINES * FAILURE_FRACTION)
    will_fail_flags = np.array([True] * n_failing + [False] * (N_MACHINES - n_failing))
    rng.shuffle(will_fail_flags)

    all_frames = []
    for machine_id in range(1, N_MACHINES + 1):
        will_fail = bool(will_fail_flags[machine_id - 1])
        all_frames.append(generate_machine_series(machine_id, rng, will_fail))

    data = pd.concat(all_frames, ignore_index=True)
    data.to_csv("data/sensor_data.csv", index=False)
    print(f"Generated {len(data)} rows across {N_MACHINES} machines "
          f"({n_failing} failing, {N_MACHINES - n_failing} healthy).")
    print("Saved to data/sensor_data.csv")


if __name__ == "__main__":
    main()
