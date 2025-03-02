import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import combinations

# 1. Generate random transactional dataset
np.random.seed(42)
items = ['A', 'B', 'C', 'D', 'E']
num_transactions = 50

# Each transaction contains 1-5 random items
transactions = []
for _ in range(num_transactions):
    num_items = np.random.randint(1, 5)
    transactions.append(set(np.random.choice(items, num_items, replace=False)))

# 2. Apriori Helper Functions
def get_frequent_itemsets(transactions, min_support=0.1):
    itemsets = {}
    for t in transactions:
        for item in t:
            itemsets[frozenset([item])] = itemsets.get(frozenset([item]), 0) + 1

    # Convert counts to support
    for itemset in itemsets:
        itemsets[itemset] /= len(transactions)

    # Filter by minimum support
    frequent_itemsets = {k: v for k, v in itemsets.items() if v >= min_support}
    return frequent_itemsets

def generate_higher_order_sets(frequent_itemsets, k):
    next_level_sets = set()
    items = set()
    for itemset in frequent_itemsets:
        items.update(itemset)

    for combo in combinations(items, k):
        next_level_sets.add(frozenset(combo))

    return next_level_sets

def calculate_support(transactions, candidate_sets):
    support_count = {c: 0 for c in candidate_sets}
    for t in transactions:
        for c in candidate_sets:
            if c.issubset(t):
                support_count[c] += 1

    # Convert to support values
    for c in support_count:
        support_count[c] /= len(transactions)

    return {k: v for k, v in support_count.items() if v >= 0.1}

# 3. Run Apriori and collect itemsets level-wise
levels = []
min_support = 0.1
frequent_itemsets = get_frequent_itemsets(transactions, min_support)
levels.append(frequent_itemsets)

k = 2
while frequent_itemsets:
    candidates = generate_higher_order_sets(frequent_itemsets, k)
    frequent_itemsets = calculate_support(transactions, candidates)
    if frequent_itemsets:
        levels.append(frequent_itemsets)
    k += 1

# 4. Prepare data for animation
itemset_sizes = []
supports = []
levels_count = []
itemset_names = []

for level_idx, level in enumerate(levels):
    for itemset, support in level.items():
        itemset_sizes.append(len(itemset))
        supports.append(support)
        levels_count.append(level_idx)
        itemset_names.append(','.join(itemset))

# 5. Create 3D Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter([], [], [], s=100)

ax.set_xlabel('Itemset Size')
ax.set_ylabel('Level (Pass)')
ax.set_zlabel('Support')
ax.set_xlim(0.5, max(itemset_sizes)+0.5)
ax.set_ylim(-0.5, len(levels)-0.5)
ax.set_zlim(0, max(supports) + 0.1)
ax.set_title("Apriori Algorithm - Frequent Itemset Evolution (3D Animation)")

# Function to update each frame
def update(frame):
    ax.clear()
    ax.set_xlabel('Itemset Size')
    ax.set_ylabel('Level (Pass)')
    ax.set_zlabel('Support')
    ax.set_title("Apriori Algorithm - Frequent Itemset Evolution (3D Animation)")

    ax.set_xlim(0.5, max(itemset_sizes)+0.5)
    ax.set_ylim(-0.5, len(levels)-0.5)
    ax.set_zlim(0, max(supports) + 0.1)

    # Draw itemsets discovered so far
    for i in range(frame+1):
        ax.scatter(itemset_sizes[i], levels_count[i], supports[i], color='teal', s=100)
        ax.text(itemset_sizes[i], levels_count[i], supports[i], itemset_names[i], fontsize=8)

ani = animation.FuncAnimation(fig, update, frames=len(itemset_sizes), interval=500, repeat=False)

# Optional - Save as GIF
ani.save('apriori_3d_animation.gif', writer='pillow', fps=2)

plt.show()
