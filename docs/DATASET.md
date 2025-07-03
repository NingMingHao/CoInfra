## Dataset Folder Structure

```bash
scenario-folder1/ (e.g., 2025_03_24_17_04_rainy/)
├── slice-folder/ (e.g., slice_0/)
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
└── ... # other slice folders

scenario-folder...
├── ...

```