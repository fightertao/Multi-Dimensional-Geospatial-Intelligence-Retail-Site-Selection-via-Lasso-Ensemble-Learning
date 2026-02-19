import matplotlib.pyplot as plt
import numpy as np

labels = ['Shopping Mall Density', 'Proximity to Malls', 'Healthcare Flux',
          'Regional Accessibility', 'Commercial Saturation']
values = [11.98, 5.72, 0.46, 0.22, -9.34]


plot_values = [abs(v) for v in values]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

plot_values += plot_values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.fill(angles, plot_values, color='#2E86C1', alpha=0.3)
ax.plot(angles, plot_values, color='#2E86C1', linewidth=2, marker='o')

for i, angle in enumerate(angles[:-1]):
    curr_angle = np.degrees(angle)
    val_str = f"{labels[i]}\n({values[i]:+.2f})"
    color = '#E74C3C' if values[i] < 0 else '#2E86C1'

    if curr_angle == 0:
        ha, va = 'center', 'bottom'
    elif 0 < curr_angle < 180:
        ha, va = 'left', 'center'
    elif curr_angle == 180:
        ha, va = 'center', 'top'
    else:
        ha, va = 'right', 'center'

    ax.text(angle, max(plot_values) * 1.2, val_str,
            color=color, size=11, fontweight='bold', ha=ha, va=va)


ax.set_yticklabels([])
ax.spines['polar'].set_visible(False)

plt.title('Site #2760: Key Growth Drivers & Risks', size=16, fontweight='bold', pad=50)
plt.tight_layout()
plt.show()