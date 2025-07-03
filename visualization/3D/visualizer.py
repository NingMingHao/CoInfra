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
import utm
from scipy.spatial.transform import Rotation as R

# ---------- Helpers for conversion ----------


def geo_to_open3d_extrinsic(x, y, z, pitch, yaw, roll):
    """
    Converts geo/Google (ENU, Z-up, yaw/pitch/roll) pose to Open3D extrinsic (world-to-cam).
    Reverse of convert_extrinsic_to_standard.
    """

    # 1. Start with Google/ENU orientation (yaw, pitch, roll) as 'zyx'
    enu_rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

    # 2. Apply +90° rotation about X to map ENU (Z-up) to Open3D (Z-forward)
    rot_x_pos90 = R.from_euler('x', 90, degrees=True).as_matrix()
    open3d_rot = enu_rot @ rot_x_pos90  # ENU to Open3D

    # Set roll to zero to match google map view
    angle_xyz = R.from_matrix(open3d_rot).as_euler('xyz', degrees=True)
    open3d_rot = R.from_euler(
        'xyz', [angle_xyz[0], 0, angle_xyz[2]], degrees=True).as_matrix()

    # 3. Build camera-to-world matrix (Google convention)
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = open3d_rot
    cam_to_world[:3, 3] = [x, y, z]

    # 4. Open3D extrinsic is world-to-cam, so invert
    world_to_cam = np.linalg.inv(cam_to_world)
    return world_to_cam


def convert_extrinsic_to_standard(open3d_extrinsic):
    """
    Convert Open3D extrinsic (world-to-cam) to standard cam-to-world (ENU/Z-up, Google convention).
    Handles the case where Open3D Z is down, Google Z is up.
    """

    # 1. Invert to get cam-to-world
    cam_to_world = np.linalg.inv(open3d_extrinsic)

    # 2. Apply rotation to convert Open3D (Z forward) to ENU/Google (Z up)
    # This is typically a -90 deg rotation about X (convert Z forward -> Z up)
    # [ [1, 0, 0], [0, 0, -1], [0, 1, 0] ] is the rot. matrix for -90 deg about X
    rot_x_neg90 = R.from_euler('x', -90, degrees=True).as_matrix()
    corrected_rot = cam_to_world[:3, :3] @ rot_x_neg90

    corrected_matrix = np.eye(4)
    corrected_matrix[:3, :3] = corrected_rot
    corrected_matrix[:3, 3] = cam_to_world[:3, 3]

    rot = R.from_matrix(corrected_matrix[:3, :3])
    # roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
    yaw, pitch, roll = rot.as_euler('zyx', degrees=True)
    x, y, z = corrected_matrix[:3, 3]

    return x, y, z, pitch, yaw, roll


def print_all_euler_conventions(matrix):
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(matrix[:3, :3])
    orders = ['zyx', 'xyz', 'yxz', 'zxy']
    for order in orders:
        try:
            angles = rot.as_euler(order, degrees=True)
            print(f"{order}: {angles}")
        except Exception as e:
            print(f"{order}: ERROR {e}")


def visualize_pointcloud_worker(base_folder, initial_timestamp, selected_nodes, node_colors, use_height_color, show_bbox, min_z, max_z, conn):
    def load_and_prepare_pcd(ts, use_height_color, min_z, max_z):
        lidar_data = load_lidar_npys(base_folder, ts, selected_nodes)
        calib = load_calibrations(base_folder)
        return stitch_pointclouds(lidar_data, calib, node_colors, use_height_color, min_z, max_z)

    pcd = load_and_prepare_pcd(initial_timestamp, use_height_color, min_z, max_z)
    ground_plane = create_xy_ground_plane(center=(-580, 530))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Pointcloud')
    # set background color to black
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().point_size = 1

    vis.add_geometry(ground_plane)
    vis.add_geometry(pcd)

    # visualize the bounding boxes
    bbox_objs = []
    # label_objs = []
    if show_bbox:
        bboxes = load_global_bboxes(base_folder, initial_timestamp)
        for obj in bboxes:
            box = create_bbox_o3d(obj)
            bbox_objs.append(box)
            vis.add_geometry(box)
            # label = create_bbox_label(obj)
            # label_objs.append(label)
            # vis.add_geometry(label)

    while True:
        vis.poll_events()
        vis.update_renderer()

        if conn.poll():
            msg = conn.recv()
            # --- Handle camera control messages
            if isinstance(msg, tuple):
                if len(msg) == 2 and isinstance(msg[0], str):
                    cmd, arg = msg
                    if cmd == 'set_camera':
                        # arg is the extrinsic (4x4 numpy array)
                        view_ctrl = vis.get_view_control()
                        params = view_ctrl.convert_to_pinhole_camera_parameters()
                        params.extrinsic = arg
                        view_ctrl.convert_from_pinhole_camera_parameters(
                            params)
                        continue
                    elif cmd == 'get_camera':
                        view_ctrl = vis.get_view_control()
                        params = view_ctrl.convert_to_pinhole_camera_parameters()
                        conn.send(params.extrinsic)

                        continue
            # --- Handle the usual point cloud update message (backward compatible)
            try:
                new_ts, use_height_color, show_bbox, min_z, max_z = msg
            except Exception as e:
                print("Unknown message format received by worker:", msg)
                continue

            view_ctrl = vis.get_view_control()
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()

            pcd_new = load_and_prepare_pcd(
                new_ts, use_height_color, min_z, max_z)
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)

            # Remove old bboxes
            for box in bbox_objs:
                vis.remove_geometry(box)
            bbox_objs.clear()

            # Add new bboxes
            if show_bbox:
                bboxes = load_global_bboxes(base_folder, new_ts)
                for obj in bboxes:
                    box = create_bbox_o3d(obj)
                    bbox_objs.append(box)
                    vis.add_geometry(box)

            # Restore previous camera
            view_ctrl.convert_from_pinhole_camera_parameters(cam_params)



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
        self.project_pcd_to_image = True
        self.min_z = 0.4
        self.max_z = 4.0

        self.detailed_image = None  # Holds the currently displayed detailed image
        self.detailed_window_open = False
        self.detailed_view_source = None  # (node_id, 'left' or 'right')

        self.ref_easting_ = 537132.0
        self.ref_northing_ = 4813391.0
        self.ref_altitude_ = 0.0  # Reference altitude for the camera
        self.camera_lat = 43.4763428
        self.camera_lon = -80.5483166
        self.camera_alt = 72.0
        self.camera_pitch = -7.38
        self.camera_yaw = 12.2
        self.camera_roll = 149.29
        self.utm_zone = 17      # Set your UTM zone and letter
        self.utm_letter = 'T'   # as appropriate for your data

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

        self.show_bbox_cb = QCheckBox("Show Bounding Boxes")
        self.show_bbox_cb.setChecked(True)
        self.show_bbox_cb.stateChanged.connect(self.toggle_show_bbox)
        z_control_layout.addWidget(self.show_bbox_cb)

        self.project_pcd_cb = QCheckBox("Project PC to Image")
        self.project_pcd_cb.setChecked(True)
        self.project_pcd_cb.stateChanged.connect(self.toggle_project_pcd)
        z_control_layout.addWidget(self.project_pcd_cb)

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

        # Camera View Controls
        cam_control_layout = QHBoxLayout()
        cam_control_layout.addWidget(QLabel("Lat:"))
        self.lat_input = QLineEdit(str(self.camera_lat))
        cam_control_layout.addWidget(self.lat_input)
        cam_control_layout.addWidget(QLabel("Lon:"))
        self.lon_input = QLineEdit(str(self.camera_lon))
        cam_control_layout.addWidget(self.lon_input)
        cam_control_layout.addWidget(QLabel("Alt (m):"))
        self.alt_input = QLineEdit(str(self.camera_alt))
        cam_control_layout.addWidget(self.alt_input)
        cam_control_layout.addWidget(QLabel("Pitch (°):"))
        self.pitch_input = QLineEdit(str(self.camera_pitch))
        cam_control_layout.addWidget(self.pitch_input)
        cam_control_layout.addWidget(QLabel("Yaw (°):"))
        self.yaw_input = QLineEdit(str(self.camera_yaw))
        cam_control_layout.addWidget(self.yaw_input)
        cam_control_layout.addWidget(QLabel("Roll (°):"))
        self.roll_input = QLineEdit(str(self.camera_roll))
        cam_control_layout.addWidget(self.roll_input)

        self.set_cam_btn = QPushButton("Set Camera View")
        self.set_cam_btn.clicked.connect(self.set_camera_view_from_ui)
        cam_control_layout.addWidget(self.set_cam_btn)

        self.get_cam_btn = QPushButton("Read Camera View")
        self.get_cam_btn.clicked.connect(self.get_camera_view_and_print)
        cam_control_layout.addWidget(self.get_cam_btn)

        layout.addLayout(cam_control_layout)

        self.setLayout(layout)

    def set_camera_view_from_ui(self):
        # Parse values from UI
        lat = float(self.lat_input.text())
        lon = float(self.lon_input.text())
        alt = float(self.alt_input.text())
        pitch = float(self.pitch_input.text())
        yaw = float(self.yaw_input.text())
        roll = float(self.roll_input.text())
        # Convert to local coordinates
        x, y, zone, letter = utm.from_latlon(lat, lon)
        x -= self.ref_easting_
        y -= self.ref_northing_
        z = alt - self.ref_altitude_  # or just alt, adjust as needed

        extrinsic = geo_to_open3d_extrinsic(x, y, z, pitch, yaw, roll)
        # Send command to visualizer process via Pipe
        if self.pcd_conn is not None:
            self.pcd_conn.send(('set_camera', extrinsic))

    def get_camera_view_and_print(self):
        # Request camera from worker and wait for reply
        if self.pcd_conn is not None:
            self.pcd_conn.send(('get_camera', None))
            if self.pcd_conn.poll(2):  # Wait for reply
                extrinsic = self.pcd_conn.recv()
                # Convert extrinsic to geo
                x, y, z, pitch, yaw, roll = convert_extrinsic_to_standard(extrinsic)
                x += self.ref_easting_
                y += self.ref_northing_
                lat, lon = utm.to_latlon(x, y, self.utm_zone, self.utm_letter)
                tmp_pitch = 35
                tmp_roll = roll - 90  # Adjust roll to match your convention
                cam_str = f"@{lat:.7f},{lon:.7f},{z + self.ref_altitude_:.2f}a,{tmp_pitch:.2f}y,{yaw:.2f}h,{tmp_roll:.2f}t"
                print(cam_str)
                # Update UI
                self.lat_input.setText(f"{lat:.7f}")
                self.lon_input.setText(f"{lon:.7f}")
                self.alt_input.setText(f"{z + self.ref_altitude_:.2f}")
                self.pitch_input.setText(f"{pitch:.2f}")
                self.yaw_input.setText(f"{yaw:.2f}")
                self.roll_input.setText(f"{roll:.2f}")
            else:
                print("No camera response from visualizer!")


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

    def toggle_project_pcd(self, state):
        self.project_pcd_to_image = state == Qt.Checked
        self.visualize_images()

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
        # Manually set the color for node 11 to avoid using gray
        self.node_colors[11] = (0.596, 0.8, 0.5)  # light yellow for node 11


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

            # img_proj = project_points_to_image(
            #     points, intrinsic, extrinsic, img.copy(), colors, alpha)
            if self.project_pcd_to_image:
                img_proj = project_points_to_image(
                    points, intrinsic, extrinsic, img.copy(), colors, alpha)
            else:
                img_proj = img.copy()
            
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
