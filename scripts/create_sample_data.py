"""Create a synthetic sample dataset for public GitHub smoke runs.

The generated files intentionally contain no private or company data. They only
match the column names and approximate tensor shapes expected by the loaders.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample"


IMU_COLS = [
    "imu_Accel_X",
    "imu_Accel_Y",
    "imu_Accel_Z",
    "imu_Gyro_X",
    "imu_Gyro_Y",
    "imu_Gyro_Z",
    "imu_Euler_X",
    "imu_Euler_Y",
    "imu_Euler_Z",
    "imu_Accel_X_delta",
    "imu_Accel_Y_delta",
    "imu_Accel_Z_delta",
    "imu_Euler_X_delta",
    "imu_Euler_Y_delta",
]
PPG_COLS = ["ppg_ppg_raw"]
SC_COLS = ["sc_scenario", "sc_scenario_type", "sc_phase"]
VEH_COLS = [
    "veh_mode",
    "veh_speed",
    "veh_accel",
    "veh_brake",
    "veh_steer",
    "veh_yaw",
    "veh_lane_offset",
    "veh_headway",
    "veh_ttc",
    "veh_collision",
    "veh_signal",
    "veh_road_type",
]
LABEL_COLS = [
    "label_motion",
    "label_tot",
    "label_valence",
    "label_arousal",
    "label_veh1_ACT",
    "label_veh2_ACT",
    "label_veh3_ACT",
]


def make_subject(pid: int, n_rows: int = 600, fs: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(20260702 + pid)
    t = np.arange(n_rows, dtype=np.float32) / fs
    df = pd.DataFrame({"timestamps": t})

    for i, col in enumerate(IMU_COLS):
        df[col] = np.sin(t * (0.6 + i * 0.03) + pid) + rng.normal(0, 0.03, n_rows)

    pulse = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 2.4 * t)
    df["ppg_ppg_raw"] = pulse + rng.normal(0, 0.02, n_rows)

    df["sc_scenario"] = (pid % 5) + 1
    df["sc_scenario_type"] = pid % 4
    phases = np.array(["before", "on", "after"], dtype=object)
    df["sc_phase"] = phases[(np.arange(n_rows) // 80) % 3]

    for i, col in enumerate(VEH_COLS):
        if col == "veh_collision":
            df[col] = ((np.arange(n_rows) + pid) % 211 == 0).astype(np.float32)
        else:
            df[col] = rng.normal(loc=i * 0.1, scale=0.5, size=n_rows).astype(np.float32)

    df["label_motion"] = (np.arange(n_rows) // 50 + pid) % 3 + 1
    df["label_tot"] = 0.4 + ((np.arange(n_rows) // 120 + pid) % 3) * 0.9
    df["label_valence"] = 5
    df["label_arousal"] = 5
    df["label_veh1_ACT"] = 1.0 + 0.5 * np.sin(t * 0.7 + pid)
    df["label_veh2_ACT"] = 1.3 + 0.4 * np.cos(t * 0.5 + pid)
    df["label_veh3_ACT"] = 1.7 + 0.3 * np.sin(t * 0.3 + pid)

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = {str(pid): make_subject(pid) for pid in range(1, 9)}

    with (OUT_DIR / "sample_train.pkl").open("wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

    survey = pd.DataFrame({"pid": list(range(1, 9))})
    for i in range(31):
        survey[f"survey_{i:02d}"] = np.linspace(0.0, 1.0, len(survey), dtype=np.float32) + i * 0.01
    survey.to_csv(OUT_DIR / "sample_survey.csv", index=False)

    print(f"Wrote {OUT_DIR / 'sample_train.pkl'}")
    print(f"Wrote {OUT_DIR / 'sample_survey.csv'}")


if __name__ == "__main__":
    main()
