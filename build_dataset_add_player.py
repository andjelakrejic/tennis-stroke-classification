import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

DATASET_DIR = "dataset"
MODEL_PATH = "pose_landmarker_full.task"

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


NEW_PLAYERS_ONLY = [
    "final_test_2"
]


def normalize_landmarks(landmarks):
    points = np.array(
        [[landmark.x, landmark.y] for landmark in landmarks],
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

    # 1. Center the skeleton around the hip center
    points = points - hip_center

    # 2. Use torso length as a measure of body size
    torso_length = np.linalg.norm(
        shoulder_center - hip_center
    )

    # Guard against division by a very small number
    if torso_length > 1e-6:
        points = points / torso_length

    return points.flatten()


def extract_sequence(video_path):
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
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
            if len(sequence) > 0:
                frame_features = sequence[-1].copy()
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
        (target_length, sequence.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(sequence.shape[1]):
        resized[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return resized


# ------------------------------------------------
# CHECK: which video files are already in the dataset
# (dedup by file path, not by player name — lets you add
# new clips for an existing player)
# ------------------------------------------------

if os.path.exists("video_paths.npy"):
    existing_video_paths = set(
        np.load("video_paths.npy", allow_pickle=True).tolist()
    )
else:
    existing_video_paths = set()


# ------------------------------------------------
# PROCESS ONLY THE NEW PLAYERS
# ------------------------------------------------

X = []
y = []
players = []
video_paths = []

for player_name in NEW_PLAYERS_ONLY:

    player_path = os.path.join(
        DATASET_DIR,
        player_name
    )

    if not os.path.isdir(player_path):
        print(f"WARNING: folder does not exist: {player_path}")
        continue

    print()
    print(f"Player: {player_name}")

    for stroke_name, label in LABELS.items():

        stroke_path = os.path.join(
            player_path,
            stroke_name
        )

        if not os.path.isdir(stroke_path):
            print(f"  WARNING: folder does not exist: {stroke_path}")
            continue

        for filename in sorted(os.listdir(stroke_path)):

            if not filename.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                continue

            video_path = os.path.join(
                stroke_path,
                filename
            )

            if video_path in existing_video_paths:
                print(f"  Already in dataset, skipping: {stroke_name}/{filename}")
                continue

            print(f"  Processing: {stroke_name}/{filename}")

            sequence = extract_sequence(
                video_path
            )

            if sequence is None:
                print("    Skipped.")
                continue

            print(
                f"    Original sequence: "
                f"{sequence.shape}"
            )

            sequence = resize_sequence(
                sequence,
                SEQUENCE_LENGTH
            )

            print(
                f"    After interpolation: "
                f"{sequence.shape}"
            )

            X.append(sequence)
            y.append(label)
            players.append(player_name)
            video_paths.append(video_path)


if len(X) == 0:
    raise RuntimeError(
        "No clips were processed. Check NEW_PLAYERS_ONLY and the folder structure."
    )

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
players = np.array(players)
video_paths = np.array(video_paths)

print()
print("NEW DATA (added players only)")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of forehands:", np.sum(y == 1))
print("Number of backhands:", np.sum(y == 0))
for player in np.unique(players):
    print(f"  {player}: {np.sum(players == player)} strokes")


# ------------------------------------------------
# MERGE WITH THE EXISTING DATASET (if it exists)
# ------------------------------------------------

if os.path.exists("X.npy"):
    print()
    print("Loading existing dataset to merge...")

    X_old = np.load("X.npy")
    y_old = np.load("y.npy")
    players_old = np.load("players.npy", allow_pickle=True)
    video_paths_old = np.load("video_paths.npy", allow_pickle=True)

    X_combined = np.concatenate([X_old, X], axis=0)
    y_combined = np.concatenate([y_old, y], axis=0)
    players_combined = np.concatenate([players_old, players], axis=0)
    video_paths_combined = np.concatenate(
        [video_paths_old, video_paths], axis=0
    )
else:
    print()
    print("No previous dataset found — creating a new one.")

    X_combined = X
    y_combined = y
    players_combined = players
    video_paths_combined = video_paths


np.save("X.npy", X_combined)
np.save("y.npy", y_combined)
np.save("players.npy", players_combined)
np.save("video_paths.npy", video_paths_combined)

print()
print("DATASET UPDATED")
print("Total clips before:", len(X_combined) - len(X))
print("New clips added:", len(X))
print("Total clips now:", len(X_combined))
print()
print("All players in the dataset:")
for player in np.unique(players_combined):
    print(f"  {player}: {np.sum(players_combined == player)} strokes")