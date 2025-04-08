import os
import open3d as o3d
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QCheckBox, QPushButton, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QLabel, QSlider)
from PyQt5.QtCore import Qt
from utils import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoInfra Visualizer")
        self.resize(1200, 600)

        self.base_folder = None
        self.timestamps = []
        self.current_timestamp = None
        self.selected_nodes = {i: True for i in range(4, 12)}

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Scenario")
        self.load_btn.clicked.connect(self.load_scenario)
        layout.addWidget(self.load_btn)

        node_layout = QHBoxLayout()
        self.node_checkboxes = {}
        for node_id in range(4, 12):
            cb = QCheckBox(f"Node {node_id}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_nodes)
            self.node_checkboxes[node_id] = cb
            node_layout.addWidget(cb)
        layout.addLayout(node_layout)

        self.slider_label = QLabel("Timestamp:")
        layout.addWidget(self.slider_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(self.slider)

        self.vis_btn = QPushButton("Visualize Pointcloud")
        self.vis_btn.clicked.connect(self.visualize_pcd)
        layout.addWidget(self.vis_btn)

        self.setLayout(layout)

    def load_scenario(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Scenario Folder")
        if folder:
            self.base_folder = folder
            ts_folder = os.path.join(folder, 'PcImage')
            self.timestamps = sorted(os.listdir(ts_folder))
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.timestamps)-1)
            self.slider.setValue(0)
            self.slider_moved()

    def slider_moved(self):
        idx = self.slider.value()
        self.current_timestamp = self.timestamps[idx]
        self.slider_label.setText(f"Timestamp: {self.current_timestamp}")

    def update_nodes(self):
        for node_id, checkbox in self.node_checkboxes.items():
            self.selected_nodes[node_id] = checkbox.isChecked()

    def create_xy_ground_plane(self, width=100, height=100, resolution=1.0, center=(0, 0)):
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

    def visualize_pcd(self):
        lidar_data = load_lidar_npys(
            self.base_folder, self.current_timestamp, self.selected_nodes)
        calib = load_calibrations(self.base_folder)

        # Generate consistent color map for nodes
        cmap = plt.get_cmap('tab10')
        node_colors = {node_id: cmap(i % 10)[:3] for i, node_id in enumerate(
            sorted(self.selected_nodes))}

        pcd = stitch_pointclouds(lidar_data, calib, node_colors)
        # Create XY ground plane
        ground_plane = self.create_xy_ground_plane(center=(-580, 530))
        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(ground_plane)
        vis.add_geometry(pcd)
        vis.run()

        # o3d.visualization.draw_geometries([pcd])

        # Visualize projection with consistent colors and transparency
        alpha = 0.6  # transparency
        

        for node_id in self.selected_nodes:
            if not self.selected_nodes[node_id]:
                continue

            for side in ['left', 'right']:
                img = load_image(self.base_folder,
                                self.current_timestamp, node_id, side)
                if img is None:
                    continue

                intrinsic = np.array(
                    calib[f"{node_id}_{side}"]["intrinsic_matrix"])
                camera_to_lidar = np.array(
                    calib[f"{node_id}_{side}"]["camera_to_lidar"])  # lidar-to-camera
                lidar_to_ground = np.array(
                    calib[f'lidar_{node_id}']['lidar_to_ground'])
                ground_to_global = np.array(
                    calib[f'lidar_{node_id}']['ground_to_global'])
                transform = ground_to_global @ lidar_to_ground @ camera_to_lidar
                extrinsic = np.linalg.inv(transform)

                # Get points in LiDAR frame (from the stitched cloud)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                # Project points to image
                img_proj = project_points_to_image(
                    points, intrinsic, extrinsic, img.copy(), colors, alpha)

                # Display image with projection
                window_name = f'Node {node_id} - {side}'
                cv2.imshow(window_name, cv2.resize(img_proj, None, fx=0.5, fy=0.5))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

