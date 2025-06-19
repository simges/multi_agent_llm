import matplotlib.pyplot as plt
import numpy as np

# https://llm-stats.com/models/compare/qwen-2.5-14b-instruct-vs-gemma-3-12b-it
# Sample data
models = ['Qwen2.5 Instruct 14B', 'Gemma3 IT 12B']
categories = ['GPQA', 'GSM8K', 'HumanEval', 'MATH', 'MBPP', 'MMLU-Pro']
results = [
    [0.455, 0.948, 0.835, 0.800, 0.820, 0.637],  # Qwen2.5 Instruct 14B
    [0.409, 0.944, 0.854, 0.838, 0.730, 0.606],  # Gemma3 IT 12B
]
colors = ['#1f77b4',  # blue
          '#ff7f0e']  # orange


# Bar chart parameters
x = np.arange(len(categories))  # the label locations
width = 0.25  # the width of the bars

# Create subplots
fig, ax = plt.subplots()

bars2 = []
# Draw each model's bar
for i in range(len(models)):
    bars = ax.bar(x + i * width, results[i], width, label=models[i], color=colors[i])
    bars2.append(bars)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, rotation=60)


# Labels and legend
ax.set_ylabel('Scores')
ax.set_title('Benchmarks')
plt.xticks(x + width, categories, rotation=45, ha='right')
ax.legend()

# Save to file
plt.tight_layout()
plt.ylim(0.2, 1.0)  # Set y-axis range from 0.5 to 1.0
plt.savefig('model_comparison.png', dpi=300)


# https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
# https://ai.google.dev/gemma/docs/core/model_card_3#benchmark_results
models2 = ['Mistral Nemo Instruct 2407', 'Gemma3 PT 12B']
categories2 = ['HellaSwag', 'WinoGrande', 'NaturalQuestions', 'TriviaQA', 'MMLU']
results2 = [
    [0.835, 0.768, 0.312, 0.738, 0.680],  # Mistral Nemo Instruct 2407
    [0.842, 0.743, 0.314, 0.782, 0.745],  # Gemma3 PT 12B
]
colors2 = ['#6a0dad',  # blue
           '#00bfc4']  # orange

x2 = np.arange(len(categories2))  # the label locations
fig2, ax2 = plt.subplots()

# Draw each model's bar
for i in range(len(models2)):
    bars = ax2.bar(x2 + i * width, results2[i], width, label=models2[i], color=colors2[i])
    bars2.append(bars)
    ax2.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, rotation=60)

# Labels and legend
ax2.set_ylabel('Scores')
ax2.set_title('Benchmarks')
plt.xticks(x2 + width, categories2, rotation=45, ha='right')
ax2.legend()

plt.tight_layout()
plt.ylim(0.2, 1.2)  # Set y-axis range from 0.5 to 1.0
plt.savefig('model_comparison2.png', dpi=300)