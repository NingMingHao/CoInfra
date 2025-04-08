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

        pcd = stitch_pointclouds(lidar_data, calib)
        print(f"Center of point cloud: {pcd.get_center()}")
        # Create XY ground plane
        ground_plane = self.create_xy_ground_plane(center=(-580, 530))
        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(ground_plane)
        vis.add_geometry(pcd)
        vis.run()

        # o3d.visualization.draw_geometries([pcd])

