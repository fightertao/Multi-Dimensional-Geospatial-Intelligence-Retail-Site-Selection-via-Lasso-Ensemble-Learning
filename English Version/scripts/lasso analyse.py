import matplotlib.pyplot as plt
import pandas as pd


data = {
    'Feature': ['Shopping Mall Density', 'Bus Station Proximity', 'Proximity to Malls',
                'General Store Density', 'Accessibility Index', 'Store Cluster Proximity',
                'Hospital Proximity', 'Hospital Density', 'Healthcare-Retail Synergy',
                'Commercial-Office Overlap'],
    'Weight': [0.4135, 0.3182, 0.2135, 0.1801, 0.0488, 0.0280, 0.0246, 0.0223, 0.0071, -0.1595]
}
weights_df = pd.DataFrame(data).sort_values(by='Weight', ascending=True)


plt.figure(figsize=(11, 8))
colors = ['#E74C3C' if x < 0 else '#2E86C1' for x in weights_df['Weight']]
bars = plt.barh(weights_df['Feature'], weights_df['Weight'], color=colors, alpha=0.9)


for bar in bars:
    w = bar.get_width()
    if w >= 0:
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}',
                 va='center', ha='left', fontsize=10, fontweight='bold')
    else:
        plt.text(w - 0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}',
                 va='center', ha='right', fontsize=10, fontweight='bold')

plt.axvline(x=0, color='black', linewidth=1.2)
plt.xlim(weights_df['Weight'].min() - 0.15, weights_df['Weight'].max() + 0.15)

plt.title('Lasso Coefficients: Key Drivers for Retail Site Selection', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Coefficient Magnitude (Weight)', fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("Lasso_Drivers_Fixed.png", dpi=300)