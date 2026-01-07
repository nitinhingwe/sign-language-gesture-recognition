import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter


# =========================
# USB Camera Settings
# =========================
CAM_INDEX = 0        # change to 1 if needed
WIDTH = 640
HEIGHT = 480
FPS = 30
USE_MJPG = True

MODEL_PATH = "gesture_model.joblib"
ENCODER_PATH = "label_encoder.joblib"

# Smoothing settings
SMOOTHING_WINDOW = 10   # how many recent predictions to vote on
MIN_CONFIDENCE = 0.55   # show prediction only if confidence >= this


model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

print("? Loaded model + label encoder")
print("Classes:", list(le.classes_))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if USE_MJPG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError(f"? Could not open camera index {CAM_INDEX}. Try 1 or check /dev/video*")

for _ in range(10):
    cap.read()

pred_history = deque(maxlen=SMOOTHING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    display_text = "No hand"
    conf_text = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        x_ = []
        y_ = []
        features = []

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            features.append(lm.x - min(x_))
            features.append(lm.y - min(y_))

        if len(features) == 42:
            X = np.array(features, dtype=np.float32).reshape(1, -1)

            pred_idx = int(model.predict(X)[0])
            pred_label = le.inverse_transform([pred_idx])[0]

            conf = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf = float(np.max(proba))

            # If we have confidence and it's too low, treat as unknown
            if conf is not None and conf < MIN_CONFIDENCE:
                pred_history.append("Unknown")
            else:
                pred_history.append(str(pred_label))

            # Majority vote
            vote = Counter(pred_history).most_common(1)[0][0]
            display_text = f"Pred: {vote}"

            if conf is not None:
                conf_text = f"Conf: {conf*100:.1f}%"

            # Draw landmarks + box
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            x_min = int(min(x_) * w)
            y_min = int(min(y_) * h)
            x_max = int(max(x_) * w)
            y_max = int(max(y_) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    else:
        pred_history.clear()

    cv2.putText(frame, display_text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if conf_text:
        cv2.putText(frame, conf_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition (USB Cam)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
