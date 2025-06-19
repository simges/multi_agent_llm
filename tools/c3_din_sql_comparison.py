import matplotlib.pyplot as plt
import numpy as np

# X-axis categories
x_labels = ['Easy', 'Medium', 'Hard', 'Extra Hard', 'All']
x = np.arange(len(x_labels))  # Numeric positions for bars

results = [ [0.927, 0.850, 0.776, 0.620, 0.819],  # C3-SQL
            [0.923, 0.874, 0.764, 0.627, 0.828],  # DIN-SQL
            #[0.927, 0.906, 0.678, 0.536, 0.813]  # Our System
]

# Bar width
width = 0.25  # the width of the bars

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0',  # Muted Blue (clear, professional)
          '#C44E52',  # Muted Red (strong, distinct)
          '#55A868'  # Soft Green (balanced, calm)
          ]

models = ['C3-SQL', 'DIN-SQL']
bars2 = []
# Plot each model's accuracy
for i, model in enumerate(models):
    bars = plt.bar(x + i * width, results[i], width, label=models[i], color=colors[i])
    bars2.append(bars)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, rotation=60)

# Axis labels and title
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Execution Accuracy')
ax.set_title('DIN-SQL, C3-SQL Comparison with Multi-Agent Setup: Execution Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

# Layout and save
plt.tight_layout()
plt.ylim(0.50, 1.0)
plt.savefig('model_3_comparison_barchart.png', dpi=300)
plt.close()

print("Chart saved as 'model_3_comparison_barchart.png'")
