import os
import yaml
import numpy as np
from glob import glob
import json
from tqdm import tqdm
from collections import defaultdict, Counter

# ========== CONFIG ==========
BASE_DIR = "/media/minghao/Data2TB/CoInfraProcessedData"
SPLIT_JSON_PATH = os.path.join(BASE_DIR, "timestamp_split.json")
SCENARIO_FOLDER_LIST = [
    "2025_02_12_16_37_heavysnow",
    "2025_03_24_17_04_rainy",
    "2025_03_27_16_53_sunny",
    "2025_04_02_17_00_freezingrain",
    "2025_04_07_17_11_heavysnow"]
NODE_IDS = [str(i) for i in range(4, 12)]

CLASS_LIST = ["Car", "Person", "Truck", "Bus", "Bicycle"]
WEATHER_MAP = {
    "heavysnow": "Heavy Snow",
    "rainy": "Rainy",
    "sunny": "Sunny",
    "freezingrain": "Freezing Rain"
}

# For polar binning
R_BIN_SIZE = 10  # meters
R_MAX = 50      # max range
THETA_BINS = 36  # 10 degree bins


def extract_weather_from_scenario(scenario_name):
    for k in WEATHER_MAP:
        if k in scenario_name:
            return WEATHER_MAP[k]
    return "Unknown"


def main():
    # Load split config
    with open(SPLIT_JSON_PATH, "r") as f:
        split_config = json.load(f)

    stats = {
        "num_lidar_frames": 0,
        "num_images": 0,
        "num_timestamps": 0,
        "objects_per_class": Counter(),
        "objects_per_weather": {weather: Counter() for weather in WEATHER_MAP.values()},
        "orientation_hist": {cls: [] for cls in CLASS_LIST},
        "distance_hist": {cls: [] for cls in CLASS_LIST},
        "polar_density": {cls: np.zeros((int(R_MAX/R_BIN_SIZE), THETA_BINS), dtype=int) for cls in CLASS_LIST},
    }

    def add_to_polar_density(cls, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)  # [-pi, pi]
        if r > R_MAX:
            return
        r_bin = int(r / R_BIN_SIZE)
        theta_bin = int(((theta + np.pi) / (2 * np.pi)) * THETA_BINS)
        if 0 <= r_bin < int(R_MAX/R_BIN_SIZE) and 0 <= theta_bin < THETA_BINS:
            stats["polar_density"][cls][r_bin, theta_bin] += 1

    for split, unit_paths in split_config.items():
        for unit_path in tqdm(unit_paths, desc=f"Analyzing {split}"):
            scenario, slice_name, timestamp = unit_path.split("/")
            weather = extract_weather_from_scenario(scenario)
            for node_id in NODE_IDS:
                gt_file = os.path.join(
                    BASE_DIR, scenario, slice_name, "GroundTruth", timestamp, f"{node_id}_roi.yaml")
                if not os.path.exists(gt_file):
                    continue
                stats["num_lidar_frames"] += 1
                # assuming 2 cameras per node per timestamp
                stats["num_images"] += 2
                stats["num_timestamps"] += 1
                try:
                    with open(gt_file, "r") as f:
                        gt_objs = yaml.safe_load(f)
                    if gt_objs is None:
                        continue
                    for obj in gt_objs:
                        cls = obj.get("label")
                        if cls not in CLASS_LIST:
                            continue
                        stats["objects_per_class"][cls] += 1
                        stats["objects_per_weather"][weather][cls] += 1
                        # Orientation stats
                        heading = obj.get("heading", 0.0)
                        stats["orientation_hist"][cls].append(heading)
                        # Distance stats
                        pos = obj.get("position", {})
                        if "x" in pos and "y" in pos:
                            distance = np.sqrt(pos["x"] ** 2 + pos["y"] ** 2 + 7.5 ** 2)  # Adding 7.5m for height
                            stats["distance_hist"][cls].append(distance)
                        else:
                            continue
                        # Polar density
                        pos = obj["position"]
                        add_to_polar_density(cls, pos["x"], pos["y"])
                except Exception as e:
                    print(f"Error reading {gt_file}: {e}")

    # Save as JSON-friendly types
    stats_save = {
        "num_lidar_frames": stats["num_lidar_frames"],
        "num_images": stats["num_images"],
        "num_timestamps": stats["num_timestamps"],
        "objects_per_class": dict(stats["objects_per_class"]),
        "objects_per_weather": {w: dict(cnt) for w, cnt in stats["objects_per_weather"].items()},
        "orientation_hist": {cls: hist for cls, hist in stats["orientation_hist"].items()},
        "distance_hist": {cls: hist for cls, hist in stats["distance_hist"].items()},
        "polar_density": {cls: pd.tolist() for cls, pd in stats["polar_density"].items()},
    }

    out_path = os.path.join(BASE_DIR, "dataset_statistics.json")
    with open(out_path, "w") as f:
        json.dump(stats_save, f, indent=2)
    print(f"Saved dataset stats to {out_path}")


if __name__ == "__main__":
    main()
