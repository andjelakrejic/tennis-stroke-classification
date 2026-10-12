import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------------------------
# PODESAVANJA
# ------------------------------------------------

TEST_DIR = "../stress_tests/angle_test/sinner"

POSE_MODEL_PATH = "../pose_landmarker_full.task"

# Ako testiras igraca5 iz novog ugla:
CLASSIFICATION_MODEL_PATH = "../tennis_stroke_lstm_test_igrac5.keras"

SEQUENCE_LENGTH = 30

NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 2
NUM_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK

LABELS = {
    "backhand": 0,
    "forehand": 1
}

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


# ------------------------------------------------
# NORMALIZACIJA LANDMARKA
# ------------------------------------------------

def normalize_landmarks(landmarks):

    points = np.array(
        [
            [landmark.x, landmark.y]
            for landmark in landmarks
        ],
        dtype=np.float32
    )

    hip_center = (
        points[LEFT_HIP] +
        points[RIGHT_HIP]
    ) / 2.0

    shoulder_center = (
        points[LEFT_SHOULDER] +
        points[RIGHT_SHOULDER]
    ) / 2.0

    # Centriranje oko centra kukova
    points = points - hip_center

    # Dužina trupa za skaliranje
    torso_length = np.linalg.norm(
        shoulder_center - hip_center
    )

    if torso_length > 1e-6:
        points = points / torso_length

    return points.flatten()


# ------------------------------------------------
# EKSTRAKCIJA SEKQUENCE IZ VIDEA
# ------------------------------------------------

def extract_sequence(video_path):

    base_options = python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.PoseLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ne mogu da otvorim: {video_path}")
        detector.close()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    sequence = []
    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            (frame_count / fps) * 1000
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            frame_features = normalize_landmarks(
                result.pose_landmarks[0]
            )

        else:

            # Ako poza nije pronađena, koristi prethodni frejm
            if len(sequence) > 0:

                frame_features = (
                    sequence[-1].copy()
                )

            else:

                frame_features = np.zeros(
                    NUM_FEATURES,
                    dtype=np.float32
                )

        sequence.append(frame_features)

        frame_count += 1

    cap.release()
    detector.close()

    if len(sequence) == 0:
        return None

    return np.array(
        sequence,
        dtype=np.float32
    )


# ------------------------------------------------
# INTERPOLACIJA NA 30 FREJMOVA
# ------------------------------------------------

def resize_sequence(sequence, target_length):

    original_length = len(sequence)

    if original_length == target_length:
        return sequence

    old_positions = np.linspace(
        0,
        1,
        original_length
    )

    new_positions = np.linspace(
        0,
        1,
        target_length
    )

    resized = np.zeros(
        (
            target_length,
            sequence.shape[1]
        ),
        dtype=np.float32
    )

    for feature_index in range(
        sequence.shape[1]
    ):

        resized[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return resized


# ------------------------------------------------
# OBRADA SAMO angle_test FOLDERA
# ------------------------------------------------

X_test = []
y_test = []
video_paths = []


for stroke_name, label in LABELS.items():

    stroke_path = os.path.join(
        TEST_DIR,
        stroke_name
    )

    if not os.path.isdir(stroke_path):
        print(
            f"Folder ne postoji: {stroke_path}"
        )
        continue

    print()
    print(f"Obrada: {stroke_name}")

    for filename in sorted(
        os.listdir(stroke_path)
    ):

        if not filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            continue

        video_path = os.path.join(
            stroke_path,
            filename
        )

        print(f"  {filename}")

        sequence = extract_sequence(
            video_path
        )

        if sequence is None:
            print("    Preskočeno.")
            continue

        sequence = resize_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        X_test.append(sequence)
        y_test.append(label)
        video_paths.append(video_path)


X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.int64
)

video_paths = np.array(
    video_paths
)


# ------------------------------------------------
# PROVERA
# ------------------------------------------------

print()
print("===================================")
print("ANGLE TEST DATASET")
print("===================================")

print("X:", X_test.shape)
print("y:", y_test.shape)

print(
    "Backhand:",
    np.sum(y_test == 0)
)

print(
    "Forehand:",
    np.sum(y_test == 1)
)


if len(X_test) == 0:
    raise RuntimeError(
        "Nema obrađenih test videa."
    )


# ------------------------------------------------
# UCITAVANJE VEC ISTRENIRANOG MODELA
# ------------------------------------------------

model = load_model(
    CLASSIFICATION_MODEL_PATH
)


# ------------------------------------------------
# EVALUACIJA
# ------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)


print()
print("===================================")
print("REZULTATI - NOVI UGAO")
print("===================================")

print(
    f"Test loss: {loss:.4f}"
)

print(
    f"Test accuracy: {accuracy:.4f}"
)


print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print()
print("Classification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "backhand",
            "forehand"
        ]
    )
)


# ------------------------------------------------
# POJEDINAČNI REZULTATI
# ------------------------------------------------

print()
print("===================================")
print("POJEDINAČNE PREDIKCIJE")
print("===================================")


for i in range(len(y_test)):

    true_name = (
        "BACKHAND"
        if y_test[i] == 0
        else "FOREHAND"
    )

    predicted_name = (
        "BACKHAND"
        if predictions[i] == 0
        else "FOREHAND"
    )

    confidence = (
        probabilities[i]
        if predictions[i] == 1
        else 1 - probabilities[i]
    )

    status = (
        "OK"
        if predictions[i] == y_test[i]
        else "GRESKA"
    )

    print(
        f"{status:6s} | "
        f"stvarno={true_name:8s} | "
        f"predikcija={predicted_name:8s} | "
        f"confidence={confidence:.3f} | "
        f"{video_paths[i]}"
    )
