import os
import numpy as np
import yaml
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def matrix_to_tf(matrix):
    trans = matrix[:3, 3]
    rot = matrix[:3, :3]
    euler = R.from_matrix(rot).as_euler(
        'xyz')[::-1]  # returns [yaw, pitch, roll]
    return {'translation': trans.tolist(), 'rotation': euler.tolist()}


def tf_to_matrix(tf_dict):
    trans = tf_dict['translation']
    euler = tf_dict['rotation']
    rot_mat = R.from_euler('xyz', euler[::-1]).as_matrix()
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = trans
    return tf_mat


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


def stitch_pointclouds(lidar_data, calib, node_colors, use_height_color, z_min=0.4, z_max=4.0):
    stitched_pcd = o3d.geometry.PointCloud()

    for idx, (node_id, points) in enumerate(sorted(lidar_data.items())):
        transform_lidar_to_ground = np.array(calib[f'lidar_{node_id}']['lidar_to_ground'])
        transform_ground_to_global = np.array(calib[f'lidar_{node_id}']['ground_to_global'])
        transform = transform_ground_to_global @ transform_lidar_to_ground

        # Transform points to global frame
        points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
        points_global = (transform @ points_hom.T).T[:, :3]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_global)

        if use_height_color:
            z = points_global[:, 2]
            z = np.clip(z, z_min, z_max)
            z_normalized = (z - z_min) / (z_max - z_min)
            colors = plt.get_cmap('rainbow')(z_normalized)[:, :3]
            pc.colors = o3d.utility.Vector3dVector(colors)
        else:
            color = node_colors[node_id]
            pc.paint_uniform_color(color)

        stitched_pcd += pc

    return stitched_pcd


def draw_colored_circles_fast(overlay, u, v, colors_valid, radius=4):
    # Step 1: Quantize to voxel grid
    voxel_size = radius * 3  # or slightly less for higher density
    u_q = (u // voxel_size).astype(np.int32)
    v_q = (v // voxel_size).astype(np.int32)

    # Step 2: Combine into 1D keys and find unique ones
    grid_keys = (v_q << 16) + u_q  # unique per voxel
    _, unique_indices = np.unique(grid_keys, return_index=True)

    # Step 3: Use unique indices to sample reduced set
    u_ds = u[unique_indices]
    v_ds = v[unique_indices]
    # reorder the color from rgb to bgr
    color_ds = (colors_valid[unique_indices] * 255).astype(np.uint8)[:, ::-1]

    # Step 4: Draw circles
    for x, y, color in zip(u_ds, v_ds, color_ds):
        cv2.circle(overlay, (x, y), radius, tuple(int(c) for c in color), -1)

    return overlay


def project_points_to_image(points, intrinsic, extrinsic, image, colors, alpha=0.7):
    overlay = image.copy()
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = extrinsic @ points_hom.T
    points_cam = points_cam[:3, :]

    valid_mask = (points_cam[2, :] > 0) & (points_cam[2, :] < 60)
    points_cam = points_cam[:, valid_mask]
    colors_cam = colors[valid_mask]

    proj_pts = intrinsic @ points_cam
    proj_pts[:2, :] /= proj_pts[2, :]

    u, v = proj_pts[0, :].astype(int), proj_pts[1, :].astype(int)
    height, width = image.shape[:2]
    valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid_pixels], v[valid_pixels]
    colors_valid = colors_cam[valid_pixels,:]

    # for x, y, color in zip(u, v, colors_valid):
    #     color = tuple((color * 255).astype(int).tolist())
    #     cv2.circle(overlay, (x, y), 4, color, -1)

    overlay = draw_colored_circles_fast(overlay, u, v, colors_valid, radius=4)

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

def create_xy_ground_plane(width=100, height=100, resolution=1.0, center=(0, 0)):
    # Generate grid points on XY plane (Z=0)
    x = np.arange(-width / 2, width / 2, resolution) + center[0]
    y = np.arange(-height / 2, height / 2, resolution) + center[1]
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # Generate triangles
    rows, cols = xx.shape
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx0 = i * cols + j
            idx1 = idx0 + 1
            idx2 = idx0 + cols
            idx3 = idx2 + 1
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals()
    return mesh
