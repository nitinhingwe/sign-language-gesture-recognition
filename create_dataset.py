import os
import pickle
from collections import Counter
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.05  # more forgiving for tricky gestures like "C"
)

DATA_DIR = "data"

data = []
labels = []
per_class_added = Counter()
per_class_failed = Counter()

for dir_ in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            per_class_failed[dir_] += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            per_class_failed[dir_] += 1
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        x_ = []
        y_ = []
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            data.append(data_aux)
            labels.append(dir_)
            per_class_added[dir_] += 1
        else:
            per_class_failed[dir_] += 1

with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("? Saved data.pickle")
print("Added per class:", dict(per_class_added))
print("Failed per class:", dict(per_class_failed))
print("Final label distribution:", dict(Counter(labels)))
