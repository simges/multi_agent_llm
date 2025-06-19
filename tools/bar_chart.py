import matplotlib.pyplot as plt
import numpy as np

# Labels
labels = [
    'concert_singer\n45 questions', 'pets_1\n42 questions', 'car_1\n92 questions', 'flight_2\n80 questions',
    'employee_hire_evaluation\n38 questions', 'cre_Doc_Template_Mgt\n84 questions', 'course_teach\n30 questions',
    'museum_visit\n18 questions', 'wta_1\n62 questions', 'battle_death\n16 questions',
    'student_transcripts_tracking\n78 questions', 'tvshow\n62 questions', 'poker_player\n40 questions',
    'voter_1\n15 questions', 'world_1\n120 questions',
    'orchestra\n40 questions', 'network_1\n56 questions', 'dog_kennels\n82 questions', 'singer\n30 questions',
    'real_estate_properties\n4 questions'
]

# Sizes for three models (example: same values, can be different)
sizes = [
    [3/45, 1/42, 23/92, 17/80, 1/38, 19/84, 2/30, 1/18, 12/62, 6/16, 18/78, 9/62, 6/40, 3/15, 43/120, 7/40, 13/56, 15/82, 3/30, 2/4],
    [6/45, 4/42, 31/92, 15/80, 0/38, 10/84, 2/30, 3/18, 7/62, 6/16, 23/78, 13/62, 0/40, 2/15, 44/120, 2/40, 10/56, 24/82, 3/30, 1/4],
    [9/45, 3/42, 29/92, 13/80, 4/38, 18/84, 10/30, 3/18, 10/62, 5/16, 24/78, 14/62, 3/40, 4/15, 54/120, 7/40, 14/56, 28/82, 5/30, 1/4],
]

# Model names and colors
model_names = ['Qwen2.5', 'Gemma3', 'Mistral Nemo']
colors = ['teal', 'orange', 'purple']

# X positions for each group
x = np.arange(len(labels))
bar_width = 0.25

# Plotting
plt.figure(figsize=(14, 6))
for i in range(3):
    plt.bar(x + i * bar_width, sizes[i], width=bar_width, label=model_names[i], color=colors[i])

# Labeling
plt.xticks(x + bar_width, labels, rotation=75, ha='right')
plt.ylabel('Fraction Correct')
plt.title('Error Rates per DB')
plt.legend()
plt.tight_layout()

# Save to file
plt.savefig("grouped_bar_chart.png", dpi=300)
# plt.show()  # Uncomment to display
