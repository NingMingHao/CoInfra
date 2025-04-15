from multiprocessing import Process, Pipe
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QCheckBox, QPushButton, QVBoxLayout, QLineEdit,
                             QHBoxLayout, QFileDialog, QLabel, QSlider, QGridLayout,
                             QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from utils import *


def visualize_pointcloud_worker(base_folder, initial_timestamp, selected_nodes, node_colors, use_height_color, show_bbox, min_z, max_z, conn):
    def load_and_prepare_pcd(ts, use_height_color, min_z, max_z):
        lidar_data = load_lidar_npys(base_folder, ts, selected_nodes)
        calib = load_calibrations(base_folder)
        return stitch_pointclouds(lidar_data, calib, node_colors, use_height_color, min_z, max_z)

    pcd = load_and_prepare_pcd(initial_timestamp, use_height_color, min_z, max_z)
    ground_plane = create_xy_ground_plane(center=(-580, 530))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Pointcloud')
    vis.add_geometry(ground_plane)
    vis.add_geometry(pcd)

    # visualize the bounding boxes
    bbox_objs = []
    label_objs = []
    if show_bbox:
        bboxes = load_global_bboxes(base_folder, initial_timestamp)
        for obj in bboxes:
            box = create_bbox_o3d(obj)
            bbox_objs.append(box)
            vis.add_geometry(box)
            label = create_bbox_label(obj)
            label_objs.append(label)
            vis.add_geometry(label)

    while True:
        vis.poll_events()
        vis.update_renderer()

        if conn.poll():
            new_ts, use_height_color, show_bbox, min_z, max_z = conn.recv()
            pcd_new = load_and_prepare_pcd(new_ts, use_height_color, min_z, max_z)
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)

            # Remove old bboxes
            for box in bbox_objs:
                vis.remove_geometry(box)
            bbox_objs.clear()
            # Remove old labels
            for label in label_objs:
                vis.remove_geometry(label)
            label_objs.clear()

            # Add new bboxes
            if show_bbox:
                bboxes = load_global_bboxes(base_folder, new_ts)
                for obj in bboxes:
                    box = create_bbox_o3d(obj)
                    bbox_objs.append(box)
                    vis.add_geometry(box)
                    label = create_bbox_label(obj)
                    label_objs.append(label)
                    vis.add_geometry(label)



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

        self.pcd_conn = None
        self.pcd_process = None
        self.use_height_color = True
        self.show_bbox = True
        self.min_z = 0.4
        self.max_z = 4.0

        self.detailed_image = None  # Holds the currently displayed detailed image
        self.detailed_window_open = False
        self.detailed_view_source = None  # (node_id, 'left' or 'right')

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

        self.show_bbox_cb = QCheckBox("Show Bounding Boxes")
        self.show_bbox_cb.setChecked(True)
        self.show_bbox_cb.stateChanged.connect(self.toggle_show_bbox)
        z_control_layout.addWidget(self.show_bbox_cb)

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
    
    def toggle_show_bbox(self, state):
        self.show_bbox = state == Qt.Checked
        self.visualize()

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
                                    args=(self.base_folder, self.current_timestamp,
                                            self.selected_nodes, self.node_colors,
                                            self.use_height_color, self.show_bbox, self.min_z, self.max_z, child_conn))
            self.pcd_process.start()
        else:
            # Send new timestamp to existing process
            if self.pcd_conn:
                self.pcd_conn.send(
                    (self.current_timestamp, self.use_height_color, self.show_bbox, self.min_z, self.max_z))

        self.visualize_images()

    def visualize_images(self):
        clear_layout(self.image_layout)
        lidar_data = load_lidar_npys(
            self.base_folder, self.current_timestamp, self.selected_nodes)
        calib = load_calibrations(self.base_folder)
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
            
            if self.show_bbox:
                global_bboxes = load_global_bboxes(
                    self.base_folder, self.current_timestamp)
                for bbox in global_bboxes:
                    img_proj = draw_projected_bbox(
                        img_proj, intrinsic, extrinsic, bbox, bbox['id'], bbox['label'])
            
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
