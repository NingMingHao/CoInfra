import os
import yaml
import numpy as np
import cv2
from matplotlib import cm
from tqdm import tqdm
from glob import glob
import json
import random
random.seed(42)
from natsort import natsorted

# ========== CONFIG ==========
BASE_DIR = "/home/minghao/Documents/Gits/OutdoorSensorNodes/CoInfra/example_data"
OUTPUT_DIR = "/media/minghao/Data2TB/OutdoorDataYOLO/ultralytics_dataset_global"

TRANS_YAML_PATH = os.path.join(BASE_DIR, "transformation.yaml")
with open(TRANS_YAML_PATH, 'r') as f:
    transformation_info = yaml.safe_load(f)
IMG_SIZE = transformation_info["img_size"]
SCALE = transformation_info["scale"]
MIN_X = transformation_info["min_x"]
MAX_Y = transformation_info["max_y"]

SPLIT_JSON_PATH = os.path.join(BASE_DIR, "timestamp_split.json")

NODE_IDS = [str(i) for i in range(4, 12)]

SCENARIO_FOLDER_LIST = [
    "2025_02_12_16_37_heavysnow",
    "2025_03_24_17_04_rainy",
    "2025_03_27_16_53_sunny",
    "2025_04_02_17_00_freezingrain",
    "2025_04_07_17_11_heavysnow"]

cached_transform_dict = {}
cached_roi_dict = {}


def to_image_coords(x, y):
    img_x = ((x - MIN_X) * SCALE).astype(np.int32)
    img_y = ((MAX_Y - y) * SCALE).astype(np.int32)
    return img_x, img_y


def render_bev(points, roi_mask):
    min_z = 0.5
    max_z = 4.0
    # first filter points by z
    points = points[(points[:, 2] >= min_z) & (points[:, 2] <= max_z)]
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    img_x, img_y = to_image_coords(x, y)

    mask = (img_x >= 0) & (img_x < IMG_SIZE) & (
        img_y >= 0) & (img_y < IMG_SIZE)
    img_x, img_y, z = img_x[mask], img_y[mask], z[mask]

    if roi_mask is not None:
        roi_mask_vals = roi_mask[img_y, img_x] > 0
        img_x, img_y, z = img_x[roi_mask_vals], img_y[roi_mask_vals], z[roi_mask_vals]

    z_norm = (z - min_z) / (max_z - min_z)
    colors = (cm.rainbow(z_norm)[:, :3] * 255).astype(np.uint8)

    for x, y, c in zip(img_x, img_y, colors):
        cv2.circle(canvas, (x, y), 2, tuple(int(i) for i in c), -1)

    return canvas


def load_transform(calib_path):
    with open(calib_path, 'r') as f:
        tf_info = yaml.safe_load(f)
    transform_lidar_to_ground = np.array(tf_info['lidar_to_ground'])
    transform_ground_to_global = np.array(tf_info['ground_to_global'])
    transform_lidar_to_global = transform_ground_to_global @ transform_lidar_to_ground
    return transform_lidar_to_global


def transform_points(points, transform):
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    points_global = (transform @ points_hom.T).T[:, :3]
    return points_global


def transform_and_export_obb(obj):
    cls_str_to_cls = {
        "Car": 0,
        "Person": 1,
        "Truck": 2,
        "Bus": 3,
        "Bicycle": 4}
    cls_str = obj.get('label')
    cls = cls_str_to_cls.get(cls_str, 0)

    pos = obj['position']
    heading = obj['heading']

    x_c, y_c = pos['x'], pos['y']
    w, l = obj['dimensions']['width'], obj['dimensions']['length']
    dx, dy = l / 2, w / 2
    corners = np.array([
        [dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [dx, -dy]
    ])
    rot = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading),  np.cos(heading)]
    ])
    rotated = (rot @ corners.T).T + np.array([x_c, y_c])

    img_x, img_y = to_image_coords(rotated[:, 0], rotated[:, 1])
    flat = np.clip(np.stack([img_x/IMG_SIZE, img_y/IMG_SIZE], axis=1).flatten(), 0, 1)
    return f"{cls} " + " ".join([f"{v:.6f}" for v in flat])

def gather_all_timestamp_units():
    units = []
    for scenario in SCENARIO_FOLDER_LIST:
        scenario_path = os.path.join(BASE_DIR, scenario)
        for slice_name in os.listdir(scenario_path):
            slice_path = os.path.join(scenario_path, slice_name)
            pc_root = os.path.join(slice_path, "PcImage")
            if not os.path.isdir(pc_root):
                continue
            for ts in os.listdir(pc_root):
                ts_path = os.path.join(pc_root, ts)
                if os.path.isdir(ts_path):
                    units.append(os.path.join(scenario, slice_name, ts))
    return units

def create_or_load_split():
    if os.path.exists(SPLIT_JSON_PATH):
        with open(SPLIT_JSON_PATH, 'r') as f:
            return json.load(f)
    units = gather_all_timestamp_units()
    random.shuffle(units)
    n = len(units)
    split = {
        "train": natsorted(units[:int(n * 0.8)]),
        "val": natsorted(units[int(n * 0.8):int(n * 0.9)]),
        "test": natsorted(units[int(n * 0.9):])
    }
    with open(SPLIT_JSON_PATH, 'w') as f:
        json.dump(split, f, indent=2)
    return split


def process_split(split_name, unit_paths):
    for unit_path in tqdm(unit_paths, desc=f"Processing {split_name}"):
        scenario, slice_name, timestamp = unit_path.split("/")
        full_slice_path = os.path.join(BASE_DIR, scenario, slice_name)
        lidar_ts_folder = os.path.join(full_slice_path, "PcImage", timestamp, "lidar")
        gt_ts_folder = os.path.join(
            full_slice_path, "GroundTruth", timestamp)
        
        # find the roi
        # roi_key = f"{scenario}/{slice_name}"
        # speed up as configs for each scenario are same
        roi_key = f"{scenario}"
        if roi_key not in cached_roi_dict:
            roi_path = os.path.join(
                BASE_DIR, scenario, slice_name, "HDmap", "ROI", "global_roi.png")
            if os.path.exists(roi_path):
                cached_roi_dict[roi_key] = cv2.imread(
                    roi_path, cv2.IMREAD_GRAYSCALE)
        roi = cached_roi_dict[roi_key]
        
        all_points = []
        for node_id in NODE_IDS:
            #transformation_key = f"{scenario}/{slice_name}/{nid}"
            # speed up as configs for each scenario are same
            transformation_key = f"{scenario}/{node_id}"
            if transformation_key not in cached_transform_dict:
                calib_path = os.path.join(
                    BASE_DIR, scenario, slice_name, "Calibration", "lidar", f"{node_id}.yaml")
                cached_transform_dict[transformation_key] = load_transform(
                    calib_path)
            transform = cached_transform_dict[transformation_key]
            
            npy_path = glob(os.path.join(
                lidar_ts_folder, f"{node_id}_*.npy"))[0]
            points = np.load(npy_path)
            transformed = transform_points(points, transform)
            all_points.append(transformed)

        if not all_points:
            print(
                f"Warning: No points found for {full_slice_path}/{timestamp}")
            continue

        all_points = np.vstack(all_points)
        bev_img = render_bev(all_points, roi)

        out_img_dir = os.path.join(
            OUTPUT_DIR, "images", split_name, scenario, slice_name)
        out_lbl_dir = os.path.join(
            OUTPUT_DIR, "labels", split_name, scenario, slice_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        cv2.imwrite(os.path.join(
            out_img_dir, timestamp+"_global.png"), bev_img)

        gt_file = os.path.join(gt_ts_folder, "global_roi.yaml")
        label_path = os.path.join(out_lbl_dir, timestamp+"_global.txt")

        if os.path.exists(gt_file):
            with open(gt_file) as f:
                data = yaml.safe_load(f)
            with open(label_path, "w") as lf:
                for obj in data:
                    try:
                        line = transform_and_export_obb(obj)
                        lf.write(line + "\n")
                    except Exception as e:
                        print(f"Warning: could not process bbox due to {e}")
                        continue
        else:
            open(label_path, "w").close()

if __name__ == "__main__":
    SPLIT_CONFIG = create_or_load_split()
    for split, units in SPLIT_CONFIG.items():
        process_split(split, units)
