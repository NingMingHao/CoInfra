import json
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
STATS_JSON_PATH = '/media/minghao/Data2TB/CoInfraProcessedData/dataset_statistics.json'
CLASS_GROUPS = {
    "Vehicle": ["Car", "Truck", "Bus"],
    "VRU": ["Person", "Bicycle"],
}

# CLASS_GROUPS = {
#     "Car": ["Car"],
#     "Truck": ["Truck"],
#     "Bus": ["Bus"],
#     "Person": ["Person"],
#     "Bicycle": ["Bicycle"],
# }

# ===== LOAD DATA =====
with open(STATS_JSON_PATH, 'r') as f:
    stats = json.load(f)

orientation_hist = stats["orientation_hist"]
distance_hist = stats["distance_hist"]

# ===== MERGE CLASSES =====
group_orient = {}
group_dist = {}
for group, cls_list in CLASS_GROUPS.items():
    all_orient = []
    all_dist = []
    for cls in cls_list:
        all_orient += orientation_hist.get(cls, [])
        all_dist += distance_hist.get(cls, [])
    group_orient[group] = np.array(all_orient, dtype=np.float32)
    group_dist[group] = np.array(all_dist, dtype=np.float32)

# ===== PLOTTING =====

fig, axes = plt.subplots(2, len(CLASS_GROUPS), figsize=(10, 8))
# ---- Orientation ----
for idx, (group, vals) in enumerate(group_orient.items()):
    # Wrap orientation to [-pi, pi]
    vals = (vals + np.pi) % (2 * np.pi) - np.pi
    ax = axes[0, idx]
    n, bins, patches = ax.hist(
        vals, bins=36, color='steelblue', alpha=0.7, edgecolor='k')
    ax.set_title(f"{group} Orientation Histogram")
    ax.set_xlabel("Heading (radian)")
    ax.set_ylabel("Count")
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])
    ax.grid(True, linestyle=':', alpha=0.5)

# ---- Distance ----
for idx, (group, vals) in enumerate(group_dist.items()):
    ax = axes[1, idx]
    ax.hist(vals, bins=30, color='darkorange', alpha=0.7, edgecolor='k')
    ax.set_title(f"{group} Distance Histogram")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
