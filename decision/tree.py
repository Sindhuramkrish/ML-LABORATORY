# ============================================
# Decision Tree using ID3 (Entropy)
# Loan Approval Prediction
# ============================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 2: Create the Loan Approval dataset
data = {
    'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old',
            'Middle', 'Young', 'Young', 'Middle'],
    'Income': ['High', 'Medium', 'High', 'Medium', 'Low', 'Low',
               'Low', 'Medium', 'High', 'Medium'],
    'Employment': ['Salaried', 'Salaried', 'Salaried', 'Self-Employed',
                   'Self-Employed', 'Salaried', 'Salaried',
                   'Self-Employed', 'Salaried', 'Self-Employed'],
    'CreditScore': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair',
                    'Excellent', 'Excellent', 'Fair', 'Excellent', 'Fair'],
    'LoanApproved': ['No', 'Yes', 'Yes', 'Yes', 'No',
                     'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)

# Step 3: Manual encoding (stable & exam-safe)
mapping = {
    'Age': {'Young': 0, 'Middle': 1, 'Old': 2},
    'Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Employment': {'Salaried': 0, 'Self-Employed': 1},
    'CreditScore': {'Fair': 0, 'Excellent': 1},
    'LoanApproved': {'No': 0, 'Yes': 1}
}

df_encoded = df.copy()
for col in mapping:
    df_encoded[col] = df[col].map(mapping[col])

print("\nEncoded Dataset:\n")
print(df_encoded)

# Step 4: Split features and target
X = df_encoded.drop('LoanApproved', axis=1)
y = df_encoded['LoanApproved']

# Step 5: Train Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Step 6: Predict loan approval for a new applicant
# Example: Young, High income, Salaried, Excellent credit
new_applicant = [[0, 2, 0, 1]]

prediction = model.predict(new_applicant)

print("\nPrediction for new applicant:")
if prediction[0] == 1:
    print("Loan Approved ✅")
else:
    print("Loan Rejected ❌")

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(18, 9))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)
plt.title("Decision Tree – Loan Approval (ID3 Algorithm)")
plt.show()