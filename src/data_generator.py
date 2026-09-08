import csv
import random
import os

os.makedirs("data", exist_ok=True)

symptoms = [
    "fever", "cough", "fatigue", "headache", "sore_throat", "runny_nose",
    "shortness_of_breath", "chest_pain", "vomiting", "diarrhea",
    "abdominal_pain", "joint_pain", "rash", "sneezing", "loss_of_smell",
    "loss_of_taste"
]

diseases_profiles = {
    "Common Cold":       {"fever":0.2, "cough":0.6, "sore_throat":0.7, "runny_nose":0.8, "sneezing":0.7},
    "Flu":               {"fever":0.9, "cough":0.6, "fatigue":0.8, "headache":0.6, "joint_pain":0.5},
    "COVID-19":          {"fever":0.8, "cough":0.7, "fatigue":0.7, "loss_of_smell":0.5, "loss_of_taste":0.5, "shortness_of_breath":0.3},
    "Gastroenteritis":   {"vomiting":0.8, "diarrhea":0.8, "abdominal_pain":0.7, "fever":0.4},
    "Malaria":           {"fever":0.95, "headache":0.6, "fatigue":0.6, "joint_pain":0.4},
    "Dengue":            {"fever":0.95, "headache":0.7, "joint_pain":0.7, "rash":0.5},
    "Migraine":          {"headache":1.0},
    "Food Poisoning":    {"vomiting":0.8, "diarrhea":0.7, "abdominal_pain":0.6},
    "Allergy":           {"sneezing":0.9, "runny_nose":0.8, "cough":0.2},
    "Bronchitis":        {"cough":0.9, "chest_pain":0.3, "shortness_of_breath":0.4, "fever":0.3}
}

for d, prof in diseases_profiles.items():
    for s in symptoms:
        if s not in prof:
            prof[s] = 0.05

rows = []
random.seed(42)
N = 3000
disease_list = list(diseases_profiles.keys())

for _ in range(N):
    disease = random.choice(disease_list)
    prof = diseases_profiles[disease]
    sample = {}
    for s in symptoms:
        p = prof.get(s, 0.05)
        val = 1 if random.random() < p else 0
        sample[s] = val
    sample["disease"] = disease
    rows.append(sample)

with open("data/dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=symptoms + ["disease"])
    writer.writeheader()
    writer.writerows(rows)

print("Generated data/dataset.csv with", N, "rows.")
