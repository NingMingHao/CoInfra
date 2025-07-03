import os
import numpy as np
import yaml
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


def load_global_bboxes(base_folder, timestamp):
    bbox_path = os.path.join(base_folder, "GroundTruth",
                             timestamp, "global.yaml")
    if not os.path.exists(bbox_path):
        return []
    with open(bbox_path, 'r') as f:
        return yaml.safe_load(f)


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
        transform_lidar_to_ground = np.array(
            calib[f'lidar_{node_id}']['lidar_to_ground'])
        transform_ground_to_global = np.array(
            calib[f'lidar_{node_id}']['ground_to_global'])
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
    colors_valid = colors_cam[valid_pixels, :]

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


def create_bbox_o3d(obj):
    # Get dimensions and transform
    l, w, h = obj['dimensions']['length'], obj['dimensions']['width'], obj['dimensions']['height']
    x, y, z = obj['position']['x'], obj['position']['y'], obj['position']['z']
    heading = obj['heading']

    # Define box geometry
    box = o3d.geometry.OrientedBoundingBox()
    box.center = np.array([x, y, z])
    box.extent = np.array([l, w, h])
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([
                                                                             0, 0, heading])
    box.R = R
    box.color = [1, 1, 1] if obj['label'] == 'Person' else [1, 1, 0]

    return box


def create_bbox_label(obj):
    x, y, z = obj['position']['x'], obj['position']['y'], obj['position']['z'] + 1.0
    text = f"{obj['id']}:{obj['label']}"
    text_3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    text_3d.translate([x, y, z])
    return text_3d


def draw_projected_bbox(image, intrinsic, extrinsic, bbox, obj_id, label):
    h_img, w_img = image.shape[:2]

    l, w, h = bbox['dimensions']['length'], bbox['dimensions']['width'], bbox['dimensions']['height']
    x, y, z = bbox['position']['x'], bbox['position']['y'], bbox['position']['z']
    heading = bbox['heading']

    # Generate 3D corners
    dx, dy, dz = l / 2, w / 2, h / 2
    corners = np.array([
        [dx,  dy,  dz], [dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [dx,  dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
    ])
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([
                                                                             0, 0, heading])
    corners = (R @ corners.T).T + np.array([x, y, z])

    # Project corners
    points_hom = np.hstack([corners, np.ones((8, 1))])
    points_cam = (extrinsic @ points_hom.T)[:3, :]
    depths = points_cam[2, :]

    # # Reject if all corners are behind or too far
    # if not np.any((depths > 3) & (depths < 60)):
    #     return image

    # Reject if any corner is behind or too far
    if np.any(depths <= 0) or np.all(depths > 60):
        return image


    proj = (intrinsic @ points_cam).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    pts_2d = proj[:, :2].astype(np.int32)

    # Count how many corners are inside image bounds
    u, v = pts_2d[:, 0], pts_2d[:, 1]
    inside = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)

    if np.count_nonzero(inside) < 2:
        return image  # Too few visible corners

    # Draw 3D box
    lines = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in lines:
        cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 255, 0), 2)

    # Draw 2D box + label
    x_min, y_min = np.min(pts_2d[:, 0]), np.min(pts_2d[:, 1])
    x_max, y_max = np.max(pts_2d[:, 0]), np.max(pts_2d[:, 1])
    if x_max < 0 or x_min >= w_img or y_max < 0 or y_min >= h_img:
        return image  # Entire box out of view

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    cv2.putText(image, f"{obj_id}:{label}", (x_min, max(0, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image
