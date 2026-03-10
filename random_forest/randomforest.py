# ==========================================
# Random Forest - Full Program
# Save All Trees + Show Voting Prediction
# ==========================================

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ==========================================
# Step 1: Load Dataset
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

print("\nDataset Loaded")
print("Classes:", class_names)

# ==========================================
# Step 2: Train Random Forest (Entropy)
# ==========================================

model = RandomForestClassifier(
    n_estimators=5,
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X, y)

print("\nRandom Forest built using ENTROPY")

# ==========================================
# Step 3: Save All Trees in One PDF
# ==========================================

with PdfPages("RandomForest_Trees.pdf") as pdf:
    
    for i, tree in enumerate(model.estimators_):
        plt.figure(figsize=(15,8))
        
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True
        )
        
        plt.title(f"Decision Tree {i+1}")
        pdf.savefig()
        plt.close()

print("All trees saved in RandomForest_Trees.pdf")

# ==========================================
# Step 4: New Data Prediction
# ==========================================

# Taking first sample as new data
new_sample = X[0].reshape(1, -1)

print("\nTree-wise Predictions:\n")

tree_predictions = []

for i, tree in enumerate(model.estimators_):
    pred = tree.predict(new_sample).astype(int)
    decoded = class_names[pred][0]
    tree_predictions.append(decoded)
    print(f"Tree {i+1} Prediction: {decoded}")

# ==========================================
# Step 5: Majority Voting
# ==========================================

final_vote = max(set(tree_predictions), key=tree_predictions.count)

print("\nFinal Prediction (Majority Voting):", final_vote)
