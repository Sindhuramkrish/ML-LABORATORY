import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

df["target"] = df["target"].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

df.to_csv("iris_dataset.csv", index=False)

X = df.drop("target", axis=1)
y = df["target"]

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

print("Sakthi Priya-2303717710422044")
def parse_input_values(raw: str):
    # accept comma-separated numbers
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} values, got {len(parts)}")
    return [float(p) for p in parts]


def get_user_input(args):
    # Priority: --input CLI -> --interactive prompt -> default sample
    if args.input:
        vals = parse_input_values(args.input)
        return vals
    if args.interactive:
        vals = []
        print("Enter feature values for a single sample:")
        for col in X.columns:
            while True:
                try:
                    v = input(f"  {col}: ")
                    vals.append(float(v))
                    break
                except ValueError:
                    print("Please enter a valid number.")
        return vals
    # If running in an interactive terminal, ask user if they'd like to input values
    if sys.stdin.isatty():
        resp = input("No input provided. Enter values now? (y/N): ")
        if resp.strip().lower().startswith('y'):
            vals = []
            print("Enter feature values for a single sample:")
            for col in X.columns:
                while True:
                    try:
                        v = input(f"  {col}: ")
                        vals.append(float(v))
                        break
                    except ValueError:
                        print("Please enter a valid number.")
            return vals

    # non-interactive or user chose not to enter values: return default sample
    return [5.1, 3.5, 1.4, 0.2]


def main():
    parser = argparse.ArgumentParser(description="Train a Decision Tree on Iris and predict a single sample.")
    parser.add_argument("--input", type=str, default=None, help="Comma-separated feature values (e.g. '5.1,3.5,1.4,0.2')")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively for feature values")
    args = parser.parse_args()

    sample = get_user_input(args)
    prediction = model.predict(pd.DataFrame([sample], columns=X.columns))
    print("Predicted Class:", prediction[0])


if __name__ == "__main__":
    main()

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=12
)
plt.savefig("iris_decision_tree.pdf", bbox_inches="tight")
plt.show()
