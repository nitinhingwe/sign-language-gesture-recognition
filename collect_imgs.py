import os
import cv2

# =========================
# USB Camera Settings
# =========================
CAM_INDEX = 0  # change to 1 if your cam is /dev/video1
WIDTH = 640
HEIGHT = 480
FPS = 30
USE_MJPG = True  # usually boosts FPS on USB cams

DATA_DIR = 'data'
NUMBER_OF_CLASSES = 3
DATASET_SIZE = 100

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if USE_MJPG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open USB camera index {CAM_INDEX}. "
        f"Try CAM_INDEX=1 or check /dev/video*"
    )

# Small warmup (some USB cams need a few frames)
for _ in range(10):
    cap.read()

for j in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')

    # Wait until user presses 'q' to start capture for this class
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror view for natural feel (optional)
        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            'Ready? Press "Q" to start capturing this class',
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f'Class: {j}',
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow('FRAME', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.imshow('FRAME', frame)

        # save image
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

        # small delay so you don't capture identical frames too fast
        cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
