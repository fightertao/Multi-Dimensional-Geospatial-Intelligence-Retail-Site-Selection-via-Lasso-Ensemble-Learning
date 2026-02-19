import matplotlib.pyplot as plt
import numpy as np
models = ['Linear', 'Lasso', 'Ridge', 'Random Forest']
train_r2 = [0.6679, 0.5754, 0.6679, 0.5811]
test_r2 = [0.4555, 0.5595, 0.4555, 0.5270]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_r2, width, label='Train $R^2$', color='#BDC3C7') # 灰色表示训练集
rects2 = ax.bar(x + width/2, test_r2, width, label='Test $R^2$', color='#2E86C1')   # 蓝色表示测试集（重点）

ax.set_ylabel('$R^2$ Score (Accuracy)', fontsize=12)
ax.set_title('Model Performance Comparison: Train vs. Test $R^2$', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 0.8)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

ax.annotate('Best Generalization\n(Lasso)',
            xy=(1.17, 0.60),
            xytext=(1.8, 0.72),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=.2",
                            color='black',
                            lw=1.5),
            fontsize=11, fontweight='bold', color='#D35400')

plt.tight_layout()
plt.savefig('model_benchmark.png', dpi=300)
plt.show()