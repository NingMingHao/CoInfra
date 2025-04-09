from multiprocessing import Process
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QCheckBox, QPushButton, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QLabel, QSlider, QGridLayout,
                             QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from utils import *


def visualize_pointcloud_worker(base_folder, timestamp, selected_nodes, node_colors, use_height_color=False):
    lidar_data = load_lidar_npys(base_folder, timestamp, selected_nodes)
    calib = load_calibrations(base_folder)
    pcd = stitch_pointclouds(lidar_data, calib, node_colors, use_height_color)
    ground_plane = create_xy_ground_plane(center=(-580, 530))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Pointcloud')
    vis.add_geometry(ground_plane)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoInfra Visualizer")
        self.resize(1400, 1000)

        self.base_folder = None
        self.timestamps = []
        self.current_timestamp = None
        self.selected_nodes = {i: True for i in range(4, 12)}
        self.node_colors = {}

        self.pcd_process = None
        self.use_height_color = True

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

        self.vis_btn = QPushButton("Visualize")
        self.vis_btn.clicked.connect(self.visualize)
        layout.addWidget(self.vis_btn)

        self.image_area = QScrollArea()
        self.image_widget = QWidget()
        self.image_layout = QGridLayout()
        self.image_widget.setLayout(self.image_layout)
        self.image_area.setWidgetResizable(True)
        self.image_area.setWidget(self.image_widget)
        layout.addWidget(self.image_area)

        self.height_color_cb = QCheckBox("Use Height Color")
        self.height_color_cb.setChecked(True)  # Default value
        self.height_color_cb.stateChanged.connect(self.toggle_height_color)
        layout.addWidget(self.height_color_cb)

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
        # check if its the current timestamp, skip update
        if self.current_timestamp == self.timestamps[idx]:
            return
        self.current_timestamp = self.timestamps[idx]
        self.slider_label.setText(f"Timestamp: {self.current_timestamp}")
        self.visualize()

    def toggle_height_color(self, state):
        self.use_height_color = state == Qt.Checked
        self.visualize()  # Optional: auto refresh when toggled

    def update_nodes(self):
        for node_id, checkbox in self.node_checkboxes.items():
            self.selected_nodes[node_id] = checkbox.isChecked()

    def visualize(self):
        cmap = plt.get_cmap('tab10')
        self.node_colors = {node_id: cmap(
            i % 10)[:3] for i, node_id in enumerate(sorted(self.selected_nodes))}
        # Terminate previous process if running
        if self.pcd_process is not None and self.pcd_process.is_alive():
            self.pcd_process.terminate()
            self.pcd_process.join()

        # Start new process
        self.pcd_process = Process(target=visualize_pointcloud_worker,
                                args=(self.base_folder, self.current_timestamp,
                                        self.selected_nodes, self.node_colors, self.use_height_color))
        self.pcd_process.start()
        self.visualize_images()


    def visualize_images(self):
        clear_layout(self.image_layout)
        lidar_data = load_lidar_npys(
            self.base_folder, self.current_timestamp, self.selected_nodes)
        calib = load_calibrations(self.base_folder)
        pcd = stitch_pointclouds(lidar_data, calib, self.node_colors, self.use_height_color)

        alpha = 0.6
        row, col = 0, 0
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
                    calib[f"{node_id}_{side}"]["camera_to_lidar"])
                lidar_to_ground = np.array(
                    calib[f'lidar_{node_id}']['lidar_to_ground'])
                ground_to_global = np.array(
                    calib[f'lidar_{node_id}']['ground_to_global'])
                transform = ground_to_global @ lidar_to_ground @ camera_to_lidar
                extrinsic = np.linalg.inv(transform)

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                img_proj = project_points_to_image(
                    points, intrinsic, extrinsic, img.copy(), colors, alpha)

                img_display = cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB)
                h, w, ch = img_display.shape
                bytes_per_line = ch * w
                qimg = QImage(img_display.data, w, h,
                              bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    320, 240, Qt.KeepAspectRatio)
                lbl = QLabel()
                lbl.setPixmap(pixmap)
                lbl.mousePressEvent = lambda e, i=img_proj: self.open_image(i)
                self.image_layout.addWidget(lbl, row, col)
                col += 1
                if col >= 4:
                    col = 0
                    row += 1

    def open_image(self, img):
        cv2.namedWindow("Detailed View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detailed View", 960, 600)
        cv2.imshow("Detailed View", img)
        # start a resizable window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
