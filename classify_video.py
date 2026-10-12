import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import find_peaks
from tensorflow.keras.models import load_model


VIDEO_PATH = "videos/igrac3/video1.mp4"

POSE_MODEL_PATH = "pose_landmarker_full.task"
CLASSIFIER_PATH = "tennis_stroke_lstm.keras"

WRIST_LANDMARK = 16

PRE_BUFFER_SECONDS = 0.8
POST_BUFFER_SECONDS = 0.4

SEQUENCE_LENGTH = 30

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

NUM_FEATURES = 66

DEBUG_CLIPS_DIR = "debug_detected_clips"
os.makedirs(DEBUG_CLIPS_DIR, exist_ok=True)

def create_detector():
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
    return vision.PoseLandmarker.create_from_options(options)


def normalize_landmarks(landmarks):
    points = np.array([[landmark.x, landmark.y] for landmark in landmarks], dtype=np.float32)

    hip_center = (
        points[LEFT_HIP] +
        points[RIGHT_HIP]
    ) / 2.0

    shoulder_center = (
        points[LEFT_SHOULDER] +
        points[RIGHT_SHOULDER]
    ) / 2.0

    points = points - hip_center

    torso_length = np.linalg.norm(shoulder_center - hip_center)
    if torso_length > 1e-6:
        points = points / torso_length

    return points.flatten()


def resize_sequence(sequence, target_length):
    original_length = len(sequence)

    if original_length == target_length:
        return sequence

    old_positions = np.linspace(0, 1, original_length)
    new_positions = np.linspace(0,1, target_length)

    resized = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)

    for feature_index in range(sequence.shape[1]):
        resized[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return resized


def detect_strokes(video_path):
    detector = create_detector()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    wrist_positions = []
    frames = []

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((frame_count / fps) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            wrist = result.pose_landmarks[0][WRIST_LANDMARK]

            wrist_positions.append((wrist.x, wrist.y))

        else:
            wrist_positions.append(
                wrist_positions[-1]
                if wrist_positions
                else (0, 0)
            )

        frame_count += 1

    cap.release()
    detector.close()

    wrist_positions = np.array(
        wrist_positions
    )

    velocity = np.linalg.norm(
        np.diff(wrist_positions, axis=0),
        axis=1
    )

    raw_peaks, _ = find_peaks(
        velocity,
        height=0.03,
        distance=int(fps * 0.5)
    )

    final_peaks = merge_nearby_peaks(
        raw_peaks,
        velocity,
        int(fps * 1.5)
    )

    return frames, fps, final_peaks


def merge_nearby_peaks(peaks, velocity, min_gap_frames):
    if len(peaks) == 0:
        return []

    groups = [[peaks[0]]]

    for p in peaks[1:]:
        first_peak_in_group = groups[-1][0]

        if p - first_peak_in_group <= min_gap_frames:
            groups[-1].append(p)
        else:
            groups.append([p])

    return [
        max(group, key=lambda idx: velocity[idx])
        for group in groups
    ]


def extract_sequence_from_frames(frames, fps):
    detector = create_detector()
    sequence = []

    for frame_index, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_index / fps) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            features = normalize_landmarks(result.pose_landmarks[0])

        else:
            if sequence:
                features = sequence[-1].copy()
            else:
                features = np.zeros(NUM_FEATURES, dtype=np.float32)

        sequence.append(features)

    detector.close()

    sequence = np.array(sequence, dtype=np.float32)
    return resize_sequence(sequence, SEQUENCE_LENGTH)


model = load_model(CLASSIFIER_PATH)

frames, fps, peaks = detect_strokes(VIDEO_PATH)

print()
print("Number of detected strokes:", len(peaks))
print()


pre_frames = int(PRE_BUFFER_SECONDS * fps)
post_frames = int(POST_BUFFER_SECONDS * fps)

for i, peak in enumerate(peaks):
    start = max(0, peak - pre_frames)
    end = min(len(frames), peak + post_frames)

    stroke_frames = frames[start:end]

    if stroke_frames:
        h, w = stroke_frames[0].shape[:2]

        debug_path = os.path.join(
            DEBUG_CLIPS_DIR,
            f"detected_{i + 1:02d}.mp4"
        )

        writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        for frame in stroke_frames:
            writer.write(frame)

        writer.release()
        print(f"Segment saved: {debug_path}")

    sequence = extract_sequence_from_frames(stroke_frames, fps)
    X = np.expand_dims(sequence, axis=0)

    probability = model.predict(X, verbose=0)[0][0]
    if probability >= 0.5:
        stroke = "FOREHAND"
        confidence = probability
    else:
        stroke = "BACKHAND"
        confidence = 1 - probability

    print(
        f"Stroke {i + 1}: "
        f"{stroke} | "
        f"confidence = {confidence:.3f} | "
        f"time = {peak / fps:.2f}s"
    )