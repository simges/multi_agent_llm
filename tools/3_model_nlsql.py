import matplotlib.pyplot as plt
import numpy as np

# Sample data
models = ['Mistral-Nemo', 'Gemma3-IT', 'Qwen2.5']
categories = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1034']

# Accuracy data (rows = models, columns = categories)
accuracy = [
    [0.690, 0.615, 0.683, 0.677, 0.698, 0.678, 0.691, 0.666, 0.676, 0.680],  # Mistral-Nemo
    [0.850, 0.685, 0.753, 0.770, 0.782, 0.773, 0.787, 0.761, 0.770, 0.767], # Gemma-it
    [0.900, 0.750, 0.773, 0.762, 0.754, 0.745, 0.746, 0.729, 0.728, 0.730],  # Qwen2.5
    [0.940, 0.845, 0.877, 0.870, 0.870, 0.838, 0.849, 0.825, 0.823, 0.819],  # Multi Agent
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(categories, accuracy[0], marker='o', label='Mistral Nemo')
plt.plot(categories, accuracy[1], marker='x', label='Gemma-it')
plt.plot(categories, accuracy[2], marker='^', label='Qwen2.5')
plt.plot(categories, accuracy[3], marker='s', label='Multi Agent')

# Customize
plt.title('Accuracy in Spider Test Benchmark')
plt.xlabel('Question-Query Pairs')
plt.ylabel('Score')
plt.ylim(0.55, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save to file
plt.savefig('model_performance_line_chart.png', dpi=300)  # Saves at high resolution
plt.close()

print("Chart saved to 'model_performance_line_chart.png'")


# Bar chart setup
bar_width = 0.25
x = np.arange(len(categories))  # [0, 1, 2, 3]

# Plot each model's accuracy
for i, model in enumerate(models):
    plt.bar(x + i * bar_width, accuracy[i], width=bar_width, label=model)

# Axis labels and ticks
plt.xlabel('Category')
plt.ylabel('Accuracy (Spider Benchmark)')
plt.title('Model Accuracy by Category')
plt.xticks(x + bar_width, categories)
plt.ylim(0, 1.0)
plt.legend()

# Save the chart
plt.tight_layout()
plt.savefig("model_accuracy_chart.png", dpi=300)
plt.close()  # Close the plot to avoid displaying in interactive sessions

print("Chart saved as model_accuracy_chart.png")