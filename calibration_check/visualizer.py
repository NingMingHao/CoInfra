from multiprocessing import Process, Pipe
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QCheckBox, QPushButton, QVBoxLayout, QLineEdit,
                             QHBoxLayout, QFileDialog, QLabel, QSlider, QGridLayout,
                             QScrollArea, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from utils import *


def visualize_pointcloud_worker(base_folder, calib, initial_timestamp, selected_nodes, node_colors, use_height_color, min_z, max_z, conn):
    def load_and_prepare_pcd(ts, calib, selected_nodes, use_height_color, min_z, max_z):
        lidar_data = load_lidar_npys(base_folder, ts, selected_nodes)
        return stitch_pointclouds(lidar_data, calib, node_colors, use_height_color, min_z, max_z)

    pcd = load_and_prepare_pcd(
        initial_timestamp, calib, selected_nodes, use_height_color, min_z, max_z)
    ground_plane = create_xy_ground_plane(center=(-580, 530))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Pointcloud')
    vis.add_geometry(ground_plane)
    vis.add_geometry(pcd)

    while True:
        vis.poll_events()
        vis.update_renderer()
        if conn.poll():  # Check if there's a new timestamp
            new_ts, calib, selected_nodes, use_height_color, min_z, max_z = conn.recv()
            pcd_new = load_and_prepare_pcd(
                new_ts, calib, selected_nodes, use_height_color, min_z, max_z)
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoInfra Visualizer")
        self.resize(1600, 1000)

        self.base_folder = None
        self.timestamps = []
        self.current_timestamp = None
        self.selected_nodes = {i: True for i in range(4, 12)}
        self.node_colors = {}

        self.pcd_conn = None
        self.pcd_process = None
        self.use_height_color = True
        self.min_z = 0.4
        self.max_z = 4.0

        self.detailed_image = None  # Holds the currently displayed detailed image
        self.detailed_window_open = False
        self.detailed_view_source = None  # (node_id, 'left' or 'right')

        self.calib = None

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

        z_control_layout = QHBoxLayout()

        self.height_color_cb = QCheckBox("Use Height Color")
        self.height_color_cb.setChecked(True)
        self.height_color_cb.stateChanged.connect(self.toggle_height_color)
        z_control_layout.addWidget(self.height_color_cb)

        z_control_layout.addWidget(QLabel("Min Z:"))
        self.min_z_input = QLineEdit(str(self.min_z))
        self.min_z_input.setFixedWidth(60)
        z_control_layout.addWidget(self.min_z_input)

        z_control_layout.addWidget(QLabel("Max Z:"))
        self.max_z_input = QLineEdit(str(self.max_z))
        self.max_z_input.setFixedWidth(60)
        z_control_layout.addWidget(self.max_z_input)

        self.update_z_btn = QPushButton("Update Z Range")
        self.update_z_btn.clicked.connect(self.update_z_range)
        z_control_layout.addWidget(self.update_z_btn)

        layout.addLayout(z_control_layout)


        # Right-side horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout, 4)

        calib_panel = QVBoxLayout()

        self.node_select = QComboBox()
        self.node_select.addItems([str(i) for i in range(4, 12)])
        calib_panel.addWidget(QLabel("Node:"))
        calib_panel.addWidget(self.node_select)

        self.calib_type_select = QComboBox()
        self.calib_type_select.addItems([
            'left_camera_to_lidar',
            'right_camera_to_lidar',
            'lidar_to_ground',
            'ground_to_global'
        ])
        calib_panel.addWidget(QLabel("Transform Type:"))
        calib_panel.addWidget(self.calib_type_select)

        self.tf_fields = {}
        for label in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
            l = QLabel(label)
            e = QLineEdit("0.0")
            calib_panel.addWidget(l)
            calib_panel.addWidget(e)
            self.tf_fields[label] = e

        self.load_tf_btn = QPushButton("Load Current Transform")
        self.load_tf_btn.clicked.connect(self.load_current_tf)
        calib_panel.addWidget(self.load_tf_btn)

        self.update_tf_btn = QPushButton("Update Calib")
        self.update_tf_btn.clicked.connect(self.update_calib_matrix)
        calib_panel.addWidget(self.update_tf_btn)

        main_layout.addLayout(calib_panel, 1)
        self.setLayout(main_layout)


    def load_scenario(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Scenario Folder")
        if folder:
            self.base_folder = folder
            self.calib = load_calibrations(folder)
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


    def load_current_tf(self):
        if not self.calib:
            return

        node = self.node_select.currentText()
        tf_key = self.calib_type_select.currentText()

        if 'camera' in tf_key:
            side = 'left' if 'left' in tf_key else 'right'
            calib_key = f"{node}_{side}"
            # remove the side from the tf_key
            tf_key = tf_key.replace(f"{side}_", "")
        else:
            calib_key = f"lidar_{node}"

        tf_matrix = np.array(self.calib[calib_key][tf_key])  # matrix from YAML

        tf = matrix_to_tf(tf_matrix)
        trans, rot = tf['translation'], tf['rotation']
        for i, key in enumerate(['x', 'y', 'z']):
            self.tf_fields[key].setText(str(round(trans[i], 6)))
        for i, key in enumerate(['yaw', 'pitch', 'roll']):
            self.tf_fields[key].setText(str(round(rot[i], 6)))

    def update_calib_matrix(self):
        node = self.node_select.currentText()
        tf_key = self.calib_type_select.currentText()

        if 'camera' in tf_key:
            side = 'left' if 'left' in tf_key else 'right'
            calib_key = f"{node}_{side}"
        else:
            calib_key = f"lidar_{node}"

        trans = [float(self.tf_fields[k].text()) for k in ['x', 'y', 'z']]
        rot = [float(self.tf_fields[k].text()) for k in ['yaw', 'pitch', 'roll']]
        tf_dict = {'translation': trans, 'rotation': rot}

        matrix = tf_to_matrix(tf_dict)
        self.calib[calib_key][tf_key] = matrix.tolist()

        print(
            f"✅ Updated: {calib_key} → {tf_key}: x y z yaw pitch roll: {' '.join(f'{v:.4f}' for v in trans + rot)}")
        self.visualize()


    def toggle_height_color(self, state):
        self.use_height_color = state == Qt.Checked
        self.visualize()  # Optional: auto refresh when toggled


    def update_z_range(self):
        try:
            print("Updating Z range, z_min:", self.min_z_input.text(), "z_max:", self.max_z_input.text())
            # check if the min_z and max_z have changed
            if self.min_z_input.text() == str(self.min_z) and self.max_z_input.text() == str(self.max_z):
                return
            else:
                self.min_z = float(self.min_z_input.text())
                self.max_z = float(self.max_z_input.text())
                # make sure min_z < max_z
                if self.min_z >= self.max_z:
                    raise ValueError("min_z should be less than max_z")
                self.visualize()
        except ValueError:
            self.min_z = 0.4
            self.max_z = 4.0


    def update_nodes(self):
        for node_id, checkbox in self.node_checkboxes.items():
            self.selected_nodes[node_id] = checkbox.isChecked()


    def visualize(self):
        cmap = plt.get_cmap('tab10')
        self.node_colors = {node_id: cmap(
            i % 10)[:3] for i, node_id in enumerate(sorted(self.selected_nodes))}

        if self.pcd_process is None or not self.pcd_process.is_alive():
            # Setup a new pipe and process if not running
            parent_conn, child_conn = Pipe()
            self.pcd_conn = parent_conn
            self.pcd_process = Process(target=visualize_pointcloud_worker,
                                    args=(self.base_folder, self.calib, self.current_timestamp,
                                            self.selected_nodes, self.node_colors,
                                            self.use_height_color, self.min_z, self.max_z, child_conn))
            self.pcd_process.start()
        else:
            # Send new timestamp to existing process
            if self.pcd_conn:
                self.pcd_conn.send((self.current_timestamp, self.calib,
                                   self.selected_nodes, self.use_height_color, self.min_z, self.max_z))

        self.visualize_images()

    def visualize_images(self):
        clear_layout(self.image_layout)
        lidar_data = load_lidar_npys(
            self.base_folder, self.current_timestamp, self.selected_nodes)
        calib = self.calib
        pcd = stitch_pointclouds(
            lidar_data, calib, self.node_colors, self.use_height_color, self.min_z, self.max_z)

        alpha = 0.6
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        def process_view_data(node_id, side):
            img = load_image(self.base_folder,
                            self.current_timestamp, node_id, side)

            if img is None:
                return None

            intrinsic = np.array(calib[f"{node_id}_{side}"]["intrinsic_matrix"])
            camera_to_lidar = np.array(
                calib[f"{node_id}_{side}"]["camera_to_lidar"])
            lidar_to_ground = np.array(
                calib[f'lidar_{node_id}']['lidar_to_ground'])
            ground_to_global = np.array(
                calib[f'lidar_{node_id}']['ground_to_global'])
            transform = ground_to_global @ lidar_to_ground @ camera_to_lidar
            extrinsic = np.linalg.inv(transform)

            img_proj = project_points_to_image(
                points, intrinsic, extrinsic, img.copy(), colors, alpha)
            
            return img_proj  # raw image only

        # Collect all tasks
        tasks = []
        for node_id in self.selected_nodes:
            if not self.selected_nodes[node_id]:
                continue
            for side in ['left', 'right']:
                tasks.append((node_id, side))

        # Run all image processing in parallel (non-GUI)
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_view_data, node_id, side): (node_id, side)
                for node_id, side in tasks
            }

            for future in futures:
                node_id, side = futures[future]
                img_proj = future.result()
                if img_proj is not None:
                    results.append((node_id, side, img_proj))

        # Now safely create Qt widgets from results (main thread)
        row, col = 0, 0
        for node_id, side, img_proj in results:
            # Update the detailed image from results if needed
            if self.detailed_window_open and self.detailed_view_source and self.detailed_view_source[0] == node_id and self.detailed_view_source[1] == side:
                self.detailed_image = img_proj
                cv2.imshow("Detailed View", img_proj)

            img_display = cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB)
            h, w, ch = img_display.shape
            bytes_per_line = ch * w
            qimg = QImage(img_display.data, w, h,
                        bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(320, 240, Qt.KeepAspectRatio)

            lbl = QLabel()
            lbl.setPixmap(pixmap)
            # lbl.mousePressEvent = lambda e, i=img_proj: self.open_image(i)
            lbl.mousePressEvent = lambda e, i=img_proj, nid=node_id, s=side: self.open_image(
                i, nid, s)


            self.image_layout.addWidget(lbl, row, col)
            col += 1
            if col >= 4:
                col = 0
                row += 1

        if self.detailed_window_open and self.detailed_image is not None:
            cv2.imshow("Detailed View", self.detailed_image)
            cv2.waitKey(1)

    def open_image(self, img, node_id, side):
        self.detailed_image = img
        self.detailed_view_source = (node_id, side)
        self.detailed_window_open = True

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
