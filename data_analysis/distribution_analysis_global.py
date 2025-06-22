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

CLASS_LIST = ["Car", "Person", "Truck", "Bus", "Bicycle"]
WEATHER_MAP = {
    "heavysnow": "Heavy Snow",
    "rainy": "Rainy",
    "sunny": "Sunny",
    "freezingrain": "Freezing Rain"
}


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
    }

    for split, unit_paths in split_config.items():
        for unit_path in tqdm(unit_paths, desc=f"Analyzing {split}"):
            scenario, slice_name, timestamp = unit_path.split("/")
            weather = extract_weather_from_scenario(scenario)
            node_id = 'global'
            gt_file = os.path.join(
                BASE_DIR, scenario, slice_name, "GroundTruth", timestamp, f"{node_id}.yaml")
            if not os.path.exists(gt_file):
                continue
            stats["num_lidar_frames"] += 8
            # assuming 2 cameras per node per timestamp
            stats["num_images"] += 16
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
                    x = obj.get("x", 0.0)
                    y = obj.get("y", 0.0)
                    r = np.sqrt(x ** 2 + y ** 2)
                    stats["distance_hist"][cls].append(r)
                    
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
    }

    out_path = os.path.join(BASE_DIR, "dataset_statistics_global.json")
    with open(out_path, "w") as f:
        json.dump(stats_save, f, indent=2)
    print(f"Saved dataset stats to {out_path}")


if __name__ == "__main__":
    main()
