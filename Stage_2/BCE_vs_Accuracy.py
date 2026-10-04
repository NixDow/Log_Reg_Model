import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set dark theme base
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 7.5), facecolor='#0d0e12')
gs = fig.add_gridspec(2, 3, height_ratios=[0.22, 1], hspace=0.35, wspace=0.15)


p_current = 0.7
acc_current = 1.0 if p_current >= 0.5 else 0.0
bce_current = -np.log(p_current)
grad_current = -1 / p_current

# Color Palette
bg_card = '#16181d'
border_color = '#272a30'
text_white = '#ffffff'
text_gray = '#9ca3af'
color_bce = '#93c5fd'      
color_acc = '#4ade80'      
color_point = '#f472b6'    

# Helper function for rounded card containers
def draw_card(ax_target):
    bbox = ax_target.get_position()
    rect = mpatches.FancyBboxPatch((bbox.x0, bbox.y0), bbox.width, bbox.height, 
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  facecolor=bg_card, edgecolor=border_color, linewidth=1.2,
                                  transform=fig.transFigure, clip_on=False, zorder=-1)
    fig.patches.append(rect)



ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(bg_card)

# Plot Data Curves
p_vals = np.linspace(0.05, 0.99, 500)
bce_vals = -np.log(p_vals)

# BCE Curve
ax.plot(p_vals, bce_vals, color=color_bce, linewidth=2.5, label=r'BCE Loss ($-\ln$)', zorder=3)

# Step Function Accuracy Line
p_acc_1 = np.linspace(0, 0.5, 100)
p_acc_2 = np.linspace(0.5, 1.0, 100)
ax.plot(p_acc_1, np.zeros_like(p_acc_1), color=color_acc, linestyle='--', linewidth=2, label='Accuracy (Step)', zorder=3)
ax.plot([0.5, 0.5], [0, 1], color=color_acc, linestyle='--', linewidth=2, zorder=3)
ax.plot(p_acc_2, np.ones_like(p_acc_2), color=color_acc, linestyle='--', linewidth=2, zorder=3)

# Scatter Highlight Points at Current p
ax.scatter([p_current], [acc_current], color=color_acc, s=90, zorder=5)
ax.scatter([p_current], [bce_current], color=color_point, s=90, zorder=5, label=r'Current $p$')

# Axes Styling
ax.set_ylim(-0.2, 3.2)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel('Predicted Probability (p)', color='#d1d5db', fontsize=11, fontweight='bold', labelpad=12)

# Added Y-Axis Label
ax.set_ylabel('Loss / Accuracy Value', color='#d1d5db', fontsize=11, fontweight='bold', labelpad=12)

ax.tick_params(colors=text_gray, labelsize=10, length=0)
ax.grid(True, axis='y', color='#272a30', linestyle='-', alpha=0.7, zorder=1)
ax.grid(False, axis='x')

# Hide bounding spines
for spine in ax.spines.values():
    spine.set_visible(False)

draw_card(ax)

# Legend at Bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, 
          fontsize=10.5, labelcolor='#d1d5db', handletextpad=0.6, columnspacing=2.5)

plt.show()