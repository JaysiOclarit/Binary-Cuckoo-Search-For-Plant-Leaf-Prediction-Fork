import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure publication typography
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.dpi'] = 300

output_dir = os.path.join('Research Paper', 'Figures')
os.makedirs(output_dir, exist_ok=True)

# 6 Subspace categories
categories = [
    'Deep Conv\nSubspace A',
    'Deep Conv\nSubspace B',
    'Bottleneck\nEmbedding',
    'Spatial Pooling\nVector',
    'Channel Weights\n(Color/Veins)',
    'Hierarchical\nMorphology'
]
N = len(categories)

# Exact retention percentages
bcs_values = [52.1, 48.3, 51.0, 49.8, 53.2, 47.9]
gbcs_values = [68.4, 65.2, 67.8, 66.1, 69.5, 64.9]

# Complete the loop for radar polygon
bcs_plot = bcs_values + bcs_values[:1]
gbcs_plot = gbcs_values + gbcs_values[:1]

# Calculate angles
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize plot with balanced square geometry
fig, ax = plt.subplots(figsize=(6.8, 7.2), subplot_kw=dict(polar=True))

# Rotate so the first axis is at the top
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set category labels with clean radial padding
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=9.5, fontweight='medium')
ax.tick_params(axis='x', pad=14)

# Configure radial grid lines and limits
ax.set_rlabel_position(30)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="#555555", size=8.5)
ax.set_ylim(0, 85)
ax.grid(color='#d0d0d0', linestyle='--', linewidth=0.8)
ax.spines['polar'].set_color('#888888')

# Plot Baseline BCS (Red dashed with square markers)
ax.plot(angles, bcs_plot, 's--', color='#d7191c', linewidth=2.0, markersize=6, 
        label='Baseline BCS (~50.4% Mean Subspace Activation)')
ax.fill(angles, bcs_plot, color='#d7191c', alpha=0.14)

# Plot Proposed GBCS (Forest Green solid with circle markers)
ax.plot(angles, gbcs_plot, 'o-', color='#1b7837', linewidth=2.4, markersize=6.5, 
        label='Proposed GBCS (~67.0% Mean Subspace Activation)')
ax.fill(angles, gbcs_plot, color='#1b7837', alpha=0.25)

# Centered Title with generous top padding
plt.title('Inception-V3 Feature Subspace Retention Profile\n(Comparing Feature Space Representational Completeness)', 
          fontweight='bold', fontsize=11.5, pad=26, y=1.04)

# Centered, perfectly balanced Legend placed directly below the radar chart
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=1, 
                   frameon=True, fancybox=True, edgecolor='#bbbbbb', fontsize=9.2, 
                   handlelength=2.5, borderpad=0.6)
legend.get_frame().set_alpha(0.95)

# Save with exact tight bounding box to prevent stretching
plt.savefig(os.path.join(output_dir, 'Fig5_Subspace_Radar_Profile.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig5_Subspace_Radar_Profile.pdf'), bbox_inches='tight')
plt.close()

print("Fig 5 successfully redesigned and saved!")
