# CoInfra
CoInfra: A Large Scale Cooperative Infrastructure Perception Dataset in Adverse Weather Conditions


# Dataset Structure:

This folder contains training data for large-scale multi-agent perception. It is organized by scenario folders, each containing sensor data, calibration information, ground truth annotations, and HD map resources.

---

## Folder Structure

```bash
train/
├── scenarios-folder/ (e.g., 2025_03_24_17_04_rainy)
│   ├── PcImage/
│   │   └── timestamp (e.g., 1742850299.2/)
│   │       ├── lidar/
│   │       │   ├── {node_id}_{exact_timestamp}.npy      # e.g., 4_1742850299.199.npy
│   │       │   └── ...
│   │       └── camera/
│   │           ├── {node_id}_{left|right}_{exact_timestamp}.jpg  # e.g., 4_left_1742850299.202.jpg
│   │           └── ...
│   │
│   ├── Calibration/
│   │   ├── lidar/
│   │   │   ├── {node_id}.yaml       # lidar-to-ground, ground-to-global
│   │   │   └── ...
│   │   └── camera/
│   │       ├── {node_id}_left.yaml   # intrinsics, camera-to-lidar
│   │       ├── {node_id}_right.yaml  # intrinsics, camera-to-lidar
│   │       └── ...
│   │
│   ├── GroundTruth/
│   │   └── timestamp (e.g., 1742850299.2/)
│   │       ├── {node_id}_roi.yaml     # object ID, class, position, heading, and dimensions in local ground frame in ROI
│   │       ├── ...
│   │       ├── global_roi.yaml    # object ID, class, position, heading, and dimensions in global frame in global ROI
│   │       └── global.yaml        # object ID, class, position, heading, and dimensions in global frame (raw data from human labeling)
│   │
│   └── HDmap/
│       ├── transformation.yaml   # pixel <-> global transformation
│       ├── hdmap.png
│       └── ROI/
│           ├── {node_id}_roi.png     # region of interest for node_id
│           ├── ...
│           └── global_roi.png        # region of interest for global frame
└── ... # other scenarios

validation/
├── ...

test/
├── ...

```

---



## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required packages using pip:

```bash
pip3 install -r requirements.txt
```
