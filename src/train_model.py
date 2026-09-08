import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

os.makedirs("models", exist_ok=True)

DATA_PATH = "data/dataset.csv"
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["disease"])
y = df["disease"]

# encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# save model and label encoder
joblib.dump(model, "models/model.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("Saved model and label encoder to models/")
