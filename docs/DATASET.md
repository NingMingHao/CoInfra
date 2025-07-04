
## Data Structure and Compression Rationale

The CoInfra dataset is organized by scenario (e.g., 2025_02_12_heavysnow) and further divided into smaller units called slices. Each slice directory contains all necessary resources for independent research, including:
* Raw sensor data (LiDAR, camera)
* Calibration files
* HD map data
* 3D bounding box annotations

To facilitate flexible and efficient downloads, the dataset is split into multiple compressed archives (~10GB per file), each corresponding to a set of slices from the same scenario. This structure allows users to:
* Download only the slices they need for their research
* Avoid downloading the entire dataset, which is over 200GB
* Manage storage space effectively
* Resume interrupted downloads without needing to re-download the entire dataset

Each compressed file is named in the following format:
```bash
[scenario]_[first-slice]-[last-slice].tar.zst
# Example: 2025_02_12_heavysnow_slice_0-slice_4.tar.zst
```

## Extraction Instructions
The archives are compressed using `zstd` for fast decompression. To extract the dataset, you need to have `zstd` installed. You can install it via package managers like `apt`, `brew`. 

**To extract an archive, use the following command:**
```bash
tar --use-compress-program=zstd -xf [archive_file.tar.zst]
```

**To extract all archives in a directory, you can use:**
```bash
for archive in *.tar.zst; do
    echo "Extracting $archive..."
    tar --use-compress-program=zstd -xf "$archive"
done
```



## Dataset Folder Structure

```bash
scenario-folder1/ (e.g., 2025_03_24_rainy/)
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