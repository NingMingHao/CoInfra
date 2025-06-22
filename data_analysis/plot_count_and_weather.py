import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 13  # adjust as needed
plt.rcParams['font.family'] = 'serif'
import json

# Input data
STATS_JSON_PATH = '/media/minghao/Data2TB/CoInfraProcessedData/dataset_statistics_global.json'

with open(STATS_JSON_PATH, 'r') as f:
    stats = json.load(f)
objects_per_weather = stats["objects_per_weather"]

class_list = ["Car", "Truck", "Bus", "Person", "Bicycle"]
weather_list = ["Sunny", "Heavy Snow", "Rainy", "Freezing Rain"]
colors = ["#DEB841", "#6BAED6", "#2878B5", "#A2AAB3"]

# Prepare data
values = np.array([[objects_per_weather[w].get(c, 0)
                  for c in class_list] for w in weather_list])
totals = values.sum(axis=0)
props = np.divide(values, totals, where=totals != 0)

# Log-scaled stacking (scale-stacked bar)
log_totals = np.log10(totals + 1)
scaled_segments = props * log_totals

x = np.arange(len(class_list))
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros_like(log_totals)

# Draw bars and percent texts
for w_idx, weather in enumerate(weather_list):
    seg = scaled_segments[w_idx]
    bars = ax.bar(
        x, seg, bottom=bottom,
        color=colors[w_idx], label=weather,
        edgecolor='black', linewidth=1.1, alpha=0.85
    )
    # Add percent text if segment is tall enough
    for i, bar in enumerate(bars):
        percent = props[w_idx, i] * 100
        height = bar.get_height()
        if height > 0.13:  # show only for visible segments; tune as needed
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[i] + height / 2,
                f"{percent:.0f}%", ha='center', va='center',
                fontsize=12, color='white', fontweight='bold'
            )
    bottom += seg

# Add total counts above bars
for i, total in enumerate(totals):
    ax.text(x[i], np.log10(total + 1) + 0.03, f"{total:,}",
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

yticks = np.array([10, 100, 1000, 10000, 100000])
yticklabels = ['10', '100', '1k', '10k', '100k']
ax.set_yticks(np.log10(yticks))
ax.set_yticklabels(yticklabels, fontsize=12)
ax.set_ylabel(
    "Total object count (log scale)\nSegment area: weather proportion (linear)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(class_list, fontsize=14)
# ax.set_title(
#     "Log-Scaled Object Counts with Linear Weather Composition", fontsize=15)
ax.legend(title="Weather", fontsize=12, title_fontsize=13,
          loc='upper right', frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
