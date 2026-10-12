import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = "dataset/ja/forehand/fh_00.mp4"

MODEL_PATH = "pose_landmarker_full.task"

NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 2
NUM_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK


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


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Greska: nije moguce otvoriti video {VIDEO_PATH}")
    detector.close()
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

sequence = []
frame_count = 0
detected_frames = 0


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

        landmarks = result.pose_landmarks[0]

        frame_features = []

        for landmark in landmarks:
            frame_features.extend([
                landmark.x,
                landmark.y
            ])

        detected_frames += 1

    else:

        # Ako MediaPipe ne pronadje pozu u trenutnom frame-u koristimo landmarke prethodnog frame-a
        if len(sequence) > 0:
            frame_features = sequence[-1].copy()

        # Ako nije detektovana poza ni u prvom frame-u privremeno postavljamo sve koordinate na 0
        else:
            frame_features = [0.0] * NUM_FEATURES

    sequence.append(frame_features)

    frame_count += 1


cap.release()
detector.close()


sequence = np.array(
    sequence,
    dtype=np.float32
)


print()
print("Obrada zavrsena.")
print("-----------------------------")
print(f"Video: {VIDEO_PATH}")
print(f"FPS: {fps:.2f}")
print(f"Ukupan broj frame-ova: {frame_count}")
print(f"Frame-ovi sa detektovanom pozom: {detected_frames}")
print(f"Broj landmarka: {NUM_LANDMARKS}")
print(f"Broj feature-a po frame-u: {NUM_FEATURES}")
print(f"Shape sekvence: {sequence.shape}")