import matplotlib.pyplot as plt
import numpy as np

# Data
# Sample data
models = ['Mistral Nemo', 'Qwen 2.5', 'Gemma 3', 'Multi Agent System']
categories = ['Easy', 'Medium', 'Hard', 'Extra Hard', 'All']
results = [
    [0.863, 0.738, 0.540, 0.398, 0.680],  # Mistral Nemo
    [0.887, 0.771, 0.655, 0.464, 0.730],  # Qwen 2.5
    [0.931, 0.814, 0.649, 0.518, 0.767],   # Gemma 3
    [0.931, 0.883, 0.724, 0.578, 0.819]   # Multi Agent
]
colors = ['#4C72B0',  # Muted Blue (clear, professional)
          '#55A868',  # Soft Green (balanced, calm)
          '#C44E52',  # Muted Red (strong, distinct)
          '#8172B2']  # Muted Purple (calm, distinct)

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
