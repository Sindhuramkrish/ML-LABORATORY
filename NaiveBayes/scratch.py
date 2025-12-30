import math
from collections import defaultdict

# -----------------------------
# Dataset (10 samples)
# -----------------------------
data = [
    ("good movie nice", "Positive"),
    ("excellent movie", "Positive"),
    ("good acting movie", "Positive"),
    ("nice acting", "Positive"),
    ("good excellent movie", "Positive"),
    ("bad movie boring", "Negative"),
    ("boring acting", "Negative"),
    ("bad acting", "Negative"),
    ("boring movie", "Negative"),
    ("bad boring movie", "Negative")
]

# -----------------------------
# Step 1: Separate data by class
# -----------------------------
classes = defaultdict(list)

for text, label in data:
    classes[label].append(text)

# -----------------------------
# Step 2: Calculate Prior Probabilities
# -----------------------------
total_docs = len(data)
priors = {}

for label in classes:
    priors[label] = len(classes[label]) / total_docs

# -----------------------------
# Step 3: Build Vocabulary and Word Counts
# -----------------------------
vocabulary = set()
word_counts = {}
total_words = {}

for label in classes:
    word_counts[label] = defaultdict(int)
    total_words[label] = 0

    for text in classes[label]:
        words = text.split()
        for word in words:
            vocabulary.add(word)
            word_counts[label][word] += 1
            total_words[label] += 1

vocab_size = len(vocabulary)

# -----------------------------
# Step 4: Conditional Probability Function
# -----------------------------
def conditional_probability(word, label):
    return (word_counts[label][word] + 1) / (total_words[label] + vocab_size)

# -----------------------------
# Step 5: Prediction Function
# -----------------------------
def predict(text):
    words = text.split()
    scores = {}

    for label in classes:
        score = math.log(priors[label])  # use log to avoid underflow
        for word in words:
            score += math.log(conditional_probability(word, label))
        scores[label] = score

    return max(scores, key=scores.get)

# -----------------------------
# Step 6: Test the Model
# -----------------------------
test_review = "good movie"
result = predict(test_review)

print("Test Review:", test_review)
print("Predicted Sentiment:", result)
