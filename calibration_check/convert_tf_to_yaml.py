import os
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

# Your config paths:
TF_BASE_NODE_DIR = "/home/minghao/Documents/Gits/OutdoorSensorNodes/OutdoorNode/src/common/transform_publisher_infra/cfg"
TF_MEC_DIR = "/home/minghao/Documents/Gits/OutdoorSensorNodes/OutdoorMEC/src/common/transform_publisher_infra/cfg/MEC1"
OUTPUT_CALIBRATION_FOLDER = "/home/minghao/Documents/Gits/OutdoorSensorNodes/CoInfra/example_data/2025_03_27_16_53_sunny/Calibration"
os.makedirs(os.path.join(OUTPUT_CALIBRATION_FOLDER, 'lidar'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_CALIBRATION_FOLDER, 'camera'), exist_ok=True)

INTERESTED_NODE_LIST = list(range(4, 12))

# ---- TF Parsing Functions (Your existing code) ----


def remove_comments(text):
    import re
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)


def read_tf_info(tf_info_path, interested_node_name):
    with open(tf_info_path, 'r') as f:
        xml_content = f.read()
    clean_xml = remove_comments(xml_content)
    root = ET.fromstring(clean_xml)
    for node in root.findall('node'):
        if node.attrib.get('name') == interested_node_name:
            args = node.attrib.get('args')
            if args is None:
                continue
            parts = args.split()[:6]
            translation = list(map(float, parts[0:3]))
            rotation = list(map(float, parts[3:6]))
            return {'translation': translation, 'rotation': rotation}
    return None


def tf_to_matrix(tf_dict):
    trans = tf_dict['translation']
    euler = tf_dict['rotation']
    rot_mat = R.from_euler('xyz', euler[::-1]).as_matrix()
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = trans
    return tf_mat


# ---- Main Conversion Logic ----
for node_id in INTERESTED_NODE_LIST:
    # For Lidar Calibration
    tf1_path = os.path.join(TF_BASE_NODE_DIR, f"node{node_id}/tf_info.launch")
    tf2_path = os.path.join(TF_MEC_DIR, "tf_info.launch")

    # Read transformations
    tf_sensor_to_ground_dict = read_tf_info(
        tf1_path, "ground_aligned_tf_broadcaster")
    tf_ground_to_global_dict = read_tf_info(
        tf2_path, f"mec_node{node_id}_tf_broadcaster")

    if tf_sensor_to_ground_dict is None or tf_ground_to_global_dict is None:
        print(f"Node {node_id}: Missing transformation, skipping.")
        continue

    T_sensor_to_ground = tf_to_matrix(tf_sensor_to_ground_dict)
    T_ground_to_global = tf_to_matrix(tf_ground_to_global_dict)

    # YAML Formatting
    yaml_data = {
        "lidar_to_ground": T_sensor_to_ground.tolist(),
        "ground_to_global": T_ground_to_global.tolist()
    }

    # Save to YAML file
    yaml_filename = os.path.join(
        OUTPUT_CALIBRATION_FOLDER, "lidar", f"{node_id}.yaml")
    with open(yaml_filename, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=None)

    print(f"Node {node_id} lidar transformation saved to {yaml_filename}")


    # For Camera Calibration
    tf1_path = os.path.join(TF_BASE_NODE_DIR, f"node{node_id}/tf_info.launch")
    camera_info_left_path = os.path.join(
        TF_BASE_NODE_DIR, f"node{node_id}/Calibration/Left/ost.yaml")
    camera_info_right_path = os.path.join(
        TF_BASE_NODE_DIR, f"node{node_id}/Calibration/Right/ost.yaml")
    # get left camera to lidar
    tf_camera_to_lidar_left_dict = read_tf_info(
        tf1_path, f"lidar_cam_tf_broadcaster_left")
    if tf_camera_to_lidar_left_dict is None:
        print(f"Node {node_id}: Missing left camera transformation, skipping.")
        continue
    T_camera_to_lidar_left = tf_to_matrix(tf_camera_to_lidar_left_dict)
    # get right camera to lidar
    tf_camera_to_lidar_right_dict = read_tf_info(
        tf1_path, f"lidar_cam_tf_broadcaster_right")
    if tf_camera_to_lidar_right_dict is None:
        print(f"Node {node_id}: Missing right camera transformation, skipping.")
        continue
    T_camera_to_lidar_right = tf_to_matrix(tf_camera_to_lidar_right_dict)
    # get intrinsic matrix
    with open(camera_info_left_path, 'r') as f:
        camera_info_left = yaml.safe_load(f)
    with open(camera_info_right_path, 'r') as f:
        camera_info_right = yaml.safe_load(f)
    # get intrinsic matrix
    intrinsic_matrix_left = np.array(camera_info_left['projection_matrix']['data']).reshape(3, 4)[:, :3]
    intrinsic_matrix_right = np.array(camera_info_right['projection_matrix']['data']).reshape(3, 4)[:, :3]

    # YAML Formatting
    left_yaml_data = {
        "camera_to_lidar": T_camera_to_lidar_left.tolist(),
        "intrinsic_matrix": intrinsic_matrix_left.tolist()
    }
    right_yaml_data = {
        "camera_to_lidar": T_camera_to_lidar_right.tolist(),
        "intrinsic_matrix": intrinsic_matrix_right.tolist()
    }
    # Save to YAML file
    left_yaml_filename = os.path.join(
        OUTPUT_CALIBRATION_FOLDER, "camera", f"{node_id}_left.yaml")
    right_yaml_filename = os.path.join(
        OUTPUT_CALIBRATION_FOLDER, "camera", f"{node_id}_right.yaml")
    with open(left_yaml_filename, 'w') as f:
        yaml.dump(left_yaml_data, f, default_flow_style=None)
    with open(right_yaml_filename, 'w') as f:
        yaml.dump(right_yaml_data, f, default_flow_style=None)
    print(f"Node {node_id} left camera transformation saved to {left_yaml_filename}")
    print(f"Node {node_id} right camera transformation saved to {right_yaml_filename}")

