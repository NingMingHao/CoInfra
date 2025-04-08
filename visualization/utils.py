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


def stitch_pointclouds(lidar_data, calib):
    stitched_pcd = o3d.geometry.PointCloud()
    cmap = plt.get_cmap('tab10')

    for idx, (node_id, points) in enumerate(lidar_data.items()):
        transform_lidar_to_ground = np.array(calib[f'lidar_{node_id}']['lidar_to_ground'])
        transform_ground_to_global = np.array(calib[f'lidar_{node_id}']['ground_to_global'])
        transform = transform_ground_to_global @ transform_lidar_to_ground
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        pc.transform(transform)

        color = cmap(idx % 10)[:3]
        pc.paint_uniform_color(color)
        stitched_pcd += pc
    return stitched_pcd


def project_points_to_image(points, intrinsic, extrinsic, image):
    P = intrinsic @ extrinsic[:3, :]
    homo_pts = np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T
    proj_pts = P @ homo_pts
    proj_pts[:2, :] /= proj_pts[2, :]
    u, v = proj_pts[:2, :]
    mask = (proj_pts[2, :] > 0) & (u >= 0) & (
        u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    for x, y in zip(u[mask].astype(int), v[mask].astype(int)):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image


def load_image(base, timestamp, node_id, side):
    img_path = os.path.join(base, 'PcImage', timestamp,
                            'camera', f'{node_id}_{side}_{timestamp}.jpg')
    return cv2.imread(img_path)
