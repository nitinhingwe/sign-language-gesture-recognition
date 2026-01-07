import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PICKLE_PATH = "data.pickle"
MODEL_OUT = "gesture_model.joblib"
LABELS_OUT = "label_encoder.joblib"

EXPECTED_LEN = 42  # 21 landmarks * (x,y)

# Load dataset
with open(DATA_PICKLE_PATH, "rb") as f:
    data_dict = pickle.load(f)

raw_data = data_dict["data"]
raw_labels = data_dict["labels"]

# Debug: see what lengths exist
lengths = [len(x) for x in raw_data]
print("Feature length distribution:", Counter(lengths))

# Filter only correct-length samples
X_list = []
y_list = []
for feat, lab in zip(raw_data, raw_labels):
    if len(feat) == EXPECTED_LEN:
        X_list.append(feat)
        y_list.append(lab)

print(f"Kept {len(X_list)} samples with length {EXPECTED_LEN}")
print(f"Removed {len(raw_data) - len(X_list)} bad samples")

if len(X_list) < 10:
    raise RuntimeError(
        "Too few valid samples after filtering. "
        "Re-run create_dataset.py after ensuring hand landmarks are detected properly."
    )

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Train model
model = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n? Test Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_))

# Save
joblib.dump(model, MODEL_OUT)
joblib.dump(le, LABELS_OUT)

print(f"\n? Saved model to: {MODEL_OUT}")
print(f"? Saved label encoder to: {LABELS_OUT}")
