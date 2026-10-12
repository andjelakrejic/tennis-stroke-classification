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

# ------------------------------------------------
# OVDE UPISI IMENA FOLDERA KOJI TREBA PONOVO
# DA SE OBRADE (mora se poklapati sa imenima u dataset/)
# ------------------------------------------------
PLAYERS_TO_REBUILD = [
    "igrac7_angle_bh",
    "igrac7_angle_fh",
]

def normalize_landmarks(landmarks):
    points = np.array(
        [[landmark.x, landmark.y] for landmark in landmarks],
        dtype=np.float32
    )

    hip_center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
    shoulder_center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0

    points = points - hip_center

    torso_length = np.linalg.norm(shoulder_center - hip_center)
    if torso_length > 1e-6:
        points = points / torso_length

    return points.flatten()


def extract_sequence(video_path):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_count / fps) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            frame_features = normalize_landmarks(result.pose_landmarks[0])
        else:
            if len(sequence) > 0:
                frame_features = sequence[-1].copy()
            else:
                frame_features = np.zeros(NUM_FEATURES, dtype=np.float32)

        sequence.append(frame_features)
        frame_count += 1

    cap.release()
    detector.close()

    if len(sequence) == 0:
        return None

    return np.array(sequence, dtype=np.float32)


def resize_sequence(sequence, target_length):
    original_length = len(sequence)
    if original_length == target_length:
        return sequence

    old_positions = np.linspace(0, 1, original_length)
    new_positions = np.linspace(0, 1, target_length)
    resized = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)

    for feature_index in range(sequence.shape[1]):
        resized[:, feature_index] = np.interp(
            new_positions, old_positions, sequence[:, feature_index]
        )

    return resized


# ------------------------------------------------
# 1. UCITAJ POSTOJECI DATASET
# ------------------------------------------------

X_old = np.load("X.npy")
y_old = np.load("y.npy")
players_old = np.load("players.npy", allow_pickle=True)
video_paths_old = np.load("video_paths.npy", allow_pickle=True)

print(f"Pre brisanja: {len(X_old)} klipova ukupno")
for player_name in PLAYERS_TO_REBUILD:
    print(f"  {player_name}: {np.sum(players_old == player_name)} postojecih klipova")

# ------------------------------------------------
# 2. UKLONI SVE POSTOJECE UNOSE ZA TE "IGRACE"
# ------------------------------------------------

keep_mask = ~np.isin(players_old, PLAYERS_TO_REBUILD)

X_kept = X_old[keep_mask]
y_kept = y_old[keep_mask]
players_kept = players_old[keep_mask]
video_paths_kept = video_paths_old[keep_mask]

# ------------------------------------------------
# 3. PONOVO OBRADI SVAKI FOLDER
# ------------------------------------------------

X_new = []
y_new = []
players_new = []
video_paths_new = []

for player_name in PLAYERS_TO_REBUILD:

    player_path = os.path.join(DATASET_DIR, player_name)

    if not os.path.isdir(player_path):
        print(f"UPOZORENJE: folder ne postoji, preskacem: {player_path}")
        continue

    print()
    print(f"Obrada foldera: {player_name}")

    for stroke_name, label in LABELS.items():
        stroke_path = os.path.join(player_path, stroke_name)

        if not os.path.isdir(stroke_path):
            print(f"  UPOZORENJE: folder ne postoji: {stroke_path}")
            continue

        for filename in sorted(os.listdir(stroke_path)):
            if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(stroke_path, filename)
            print(f"  Processing: {stroke_name}/{filename}")

            sequence = extract_sequence(video_path)

            if sequence is None:
                print("    Skipped.")
                continue

            sequence = resize_sequence(sequence, SEQUENCE_LENGTH)

            X_new.append(sequence)
            y_new.append(label)
            players_new.append(player_name)
            video_paths_new.append(video_path)

if len(X_new) == 0:
    raise RuntimeError("Nije obradjen nijedan klip — proveri PLAYERS_TO_REBUILD i putanje.")

X_new = np.array(X_new, dtype=np.float32)
y_new = np.array(y_new, dtype=np.int64)
players_new = np.array(players_new)
video_paths_new = np.array(video_paths_new)

print()
print("Novo obradjeno ukupno:", len(X_new))
for player_name in PLAYERS_TO_REBUILD:
    mask = players_new == player_name
    print(f"  {player_name}: {mask.sum()} klipova "
          f"({np.sum(y_new[mask] == 0)} bh, {np.sum(y_new[mask] == 1)} fh)")

print()
print("Spisak svih obradjenih fajlova (proveri da nema duplikata):")
for vp in video_paths_new:
    print(f"  {vp}")

# ------------------------------------------------
# 4. SPOJI OCUVANE PODATKE SA NOVO OBRADJENIM
# ------------------------------------------------

X_combined = np.concatenate([X_kept, X_new], axis=0)
y_combined = np.concatenate([y_kept, y_new], axis=0)
players_combined = np.concatenate([players_kept, players_new], axis=0)
video_paths_combined = np.concatenate([video_paths_kept, video_paths_new], axis=0)

np.save("X.npy", X_combined)
np.save("y.npy", y_combined)
np.save("players.npy", players_combined)
np.save("video_paths.npy", video_paths_combined)

print()
print("===================================")
print("GOTOVO")
print("===================================")
print(f"Ukupno klipova sada: {len(X_combined)}")
for player_name in PLAYERS_TO_REBUILD:
    print(f"  {player_name}: {np.sum(players_combined == player_name)} klipova")