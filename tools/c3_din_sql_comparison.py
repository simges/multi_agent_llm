import matplotlib.pyplot as plt
import numpy as np

# X-axis categories
x_labels = ['Easy', 'Medium', 'Hard', 'Extra Hard', 'All']
x = np.arange(len(x_labels))  # Numeric positions for bars

# C3-SQL
model1_y1 = [0.931, 0.845, 0.799, 0.639, 0.825]  # Execution Accuracy

# DIN-SQL
model2_y1 = [0.935, 0.870, 0.799, 0.651, 0.838]  # Execution Accuracy

# Our System
model3_y1 = [0.931, 0.883, 0.724, 0.578, 0.819]  # Execution Accuracy


# Bar width
width = 0.2

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0',  # Muted Blue (clear, professional)
          '#55A868',  # Soft Green (balanced, calm)
          '#C44E52']  # Muted Red (strong, distinct)

# Plot bars
bars1 = ax.bar(x - width, model1_y1, width, label='C3-SQL', color=colors[0])
bars2 = ax.bar(x, model2_y1, width, label='DIN-SQL', color=colors[1])
bars3 = ax.bar(x + width, model3_y1, width, label='Multi Agent System', color=colors[2])

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
