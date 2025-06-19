import matplotlib.pyplot as plt
import numpy as np

# Data
# Sample data
models = ['Phi4', 'Qwen2.5 Instruct 14B', 'Gemma3 IT 12B']
categories = ['Easy', 'Medium', 'Hard', 'Extra Hard', 'All']
results = [
    [0.867, 0.749, 0.569, 0.361, 0.685],  # Mistral Nemo
    [0.899, 0.800, 0.649, 0.530, 0.755],  # Qwen 2.5 14b instruct
    [0.919, 0.803, 0.672, 0.476, 0.756],   # Gemma 3
    #[0.931, 0.883, 0.724, 0.578, 0.819]   # Multi Agent
]
colors = ['#4C72B0',  # Muted Blue (clear, professional)
          '#55A868',  # Soft Green (balanced, calm)
          '#C44E52']  # Muted Purple (calm, distinct)

# Setup
x = np.arange(len(categories))  # Category positions
bar_width = 0.15

# Plotting
plt.figure(figsize=(8, 5))
for i in range(len(models)):
    plt.bar(x + i * bar_width, results[i], width=bar_width, label=models[i], color=colors[i])

# Axes and labels
plt.xticks(x + bar_width, categories)
plt.ylabel('Accuracy')
plt.title('Model Accuracy by Difficulty Level')
plt.ylim(0.3, 1.0)  # Set y-axis range from 0.5 to 1.0
plt.legend()
plt.tight_layout()

# Save to file
plt.savefig('model_accuracy_bar_chart_limited.png', dpi=300)
# plt.show()  # Uncomment to display
