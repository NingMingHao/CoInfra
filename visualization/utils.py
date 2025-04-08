import os
import numpy as np
import yaml
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


def load_lidar_npys(base, timestamp, selected_nodes):
    lidar_path = os.path.join(base, 'PcImage', timestamp, 'lidar')
    lidar_data = {}
    for file in os.listdir(lidar_path):
        node_id = int(file.split('_')[0])
        if selected_nodes.get(node_id, False):
            points = np.load(os.path.join(lidar_path, file))
            lidar_data[node_id] = points
    return lidar_data


def load_calibrations(base):
    calib = {}
    calib_path = os.path.join(base, 'Calibration')
    for node_id in range(4, 12):
        lidar_yaml = os.path.join(calib_path, 'lidar', f'{node_id}.yaml')
        with open(lidar_yaml) as f:
            calib[f'lidar_{node_id}'] = yaml.safe_load(f)
        for side in ['left', 'right']:
            cam_yaml = os.path.join(
                calib_path, 'camera', f'{node_id}_{side}.yaml')
            with open(cam_yaml) as f:
                calib[f'{node_id}_{side}'] = yaml.safe_load(f)
    return calib


def stitch_pointclouds(lidar_data, calib, node_colors):
    stitched_pcd = o3d.geometry.PointCloud()

    for idx, (node_id, points) in enumerate(sorted(lidar_data.items())):
        transform_lidar_to_ground = np.array(calib[f'lidar_{node_id}']['lidar_to_ground'])
        transform_ground_to_global = np.array(calib[f'lidar_{node_id}']['ground_to_global'])
        transform = transform_ground_to_global @ transform_lidar_to_ground
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        pc.transform(transform)
        color = node_colors[node_id]
        pc.paint_uniform_color(color)
        stitched_pcd += pc
    return stitched_pcd


def project_points_to_image(points, intrinsic, extrinsic, image, colors, alpha=0.6):
    overlay = image.copy()
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = extrinsic @ points_hom.T
    points_cam = points_cam[:3, :]

    valid_mask = points_cam[2, :] > 0
    points_cam = points_cam[:, valid_mask]
    colors_cam = colors[valid_mask]

    proj_pts = intrinsic @ points_cam
    proj_pts[:2, :] /= proj_pts[2, :]

    u, v = proj_pts[0, :].astype(int), proj_pts[1, :].astype(int)
    height, width = image.shape[:2]
    valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid_pixels], v[valid_pixels]
    colors_valid = colors_cam[valid_pixels,:]

    for x, y, color in zip(u, v, colors_valid):
        color = tuple((color * 255).astype(int).tolist())
        cv2.circle(overlay, (x, y), 3, color, -1)

    # Blend original image with overlay
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended


def load_image(base, timestamp, node_id, side):
    img_folder_path = os.path.join(base, 'PcImage', timestamp, 'camera')
    # match the one starting with {node_id}_{side}
    # img_path = os.path.join(base, 'PcImage', timestamp,
    #                         'camera', f'{node_id}_{side}_{timestamp}.jpg')
    for file in os.listdir(img_folder_path):
        if file.startswith(f"{node_id}_{side}"):
            return cv2.imread(os.path.join(img_folder_path, file))
