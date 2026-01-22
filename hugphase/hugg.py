
import math
from transformers import pipeline
# Dataset
satisfied = [
    "service was quick and helpful",
    "happy with customer support",
    "support team was polite",
    "good service experience",
    "problem solved quickly"
]
unsatisfied = [
    "very slow response",
    "issue not resolved",
    "poor customer care",
    "worst support ever",
    "not helpful at all"
]

# Word count
def word_count(data):
    d = {}
    for s in data:
        for w in s.split():
            d[w] = d.get(w, 0) + 1

    return d
sat_words = word_count(satisfied)
unsat_words = word_count(unsatisfied)
# Priors
total = len(satisfied) + len(unsatisfied)
p_sat = len(satisfied) / total
p_unsat = len(unsatisfied) / total

vocab = set(sat_words) | set(unsat_words)
V = len(vocab)
sat_total = sum(sat_words.values())
unsat_total = sum(unsat_words.values()
# Naive Bayes
def predict_nb(text):
    s_score = math.log(p_sat)
    u_score = math.log(p_unsat)
    for w in text.split():
        s_score += math.log((sat_words.get(w, 0) + 1) / (sat_total + V))
        u_score += math.log((unsat_words.get(w, 0) + 1) / (unsat_total + V))
    return "Satisfied" if s_score > u_score else "Unsatisfied"

# Test data
tests = [
    "quick and helpful service",
    "poor customer support",
    "support team was polite",
    "very slow response",
    "not helpful at all",
    "happy with the service"
]

print("Naive Bayes:\n")
for t in tests:
    print(f"{t} -> {predict_nb(t)}")

# Hugging Face model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("\nHugging Face:\n")
or t in tests:
    r = classifier(t)[0]
    label = "Satisfied" if r["label"] == "POSITIVE" else "Unsatisfied"
    print(f"{t} -> {label} ({r['score']:.2f})")

# Comparison
print("\nComparison:")
print("Feedback".ljust(30), "NB".ljust(12), "HF")
for t in tests:
    nb = predict_nb(t)
    hf = "Satisfied" if classifier(t)[0]["label"] == "POSITIVE" else "Unsatisfied"
  print(t.ljust(30), nb.ljust(12), hf)
