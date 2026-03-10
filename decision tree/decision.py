import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    "Contains_Offer": ["Yes","Yes","Yes","No","No","Yes","No","No","Yes","No"],
    "Contains_Free":  ["Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes"],
    "Sender_Known":   ["No","Yes","No","No","Yes","Yes","Yes","No","No","No"],
    "Spam":           ["Yes","No","Yes","Yes","No","No","No","No","Yes","Yes"]
}

df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)

mapping = {
    "Contains_Offer": {"No": 0, "Yes": 1},
    "Contains_Free":  {"No": 0, "Yes": 1},
    "Sender_Known":   {"No": 0, "Yes": 1},
    "Spam":           {"No": 0, "Yes": 1}
}

df_encoded = df.copy()
for col in mapping:
    df_encoded[col] = df[col].map(mapping[col])

print("\nEncoded Dataset:\n")
print(df_encoded)

X = df_encoded.drop("Spam", axis=1)
y = df_encoded["Spam"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

new_email = [[1, 1, 0]]
prediction = model.predict(new_email)

print("\nPrediction for new email:")
if prediction[0] == 1:
    print("📩 This email is SPAM ❌")
else:
    print("📩 This email is NOT SPAM ✅")

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Spam", "Spam"],
    filled=True,
    fontsize=14
)
plt.title("Decision Tree – Spam Detection (ID3 Algorithm)", fontsize=16)
plt.savefig("spam_decision_tree.pdf", bbox_inches="tight")
plt.show()

print("\nDecision Tree PDF saved as: spam_decision_tree.pdf")
