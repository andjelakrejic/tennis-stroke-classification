# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
# MODEL_PATH = "pose_landmarker_full.task"
#
#
# def make_detector():
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.VIDEO,
#         min_pose_detection_confidence=0.5,
#         min_pose_presence_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     return vision.PoseLandmarker.create_from_options(options)
#
#
# def analyze_dominant_hand(video_path):
#     detector = make_detector()
#
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     frame_count = 0
#
#     left_positions = []    # landmark 15 - levi ručni zglob
#     right_positions = []   # landmark 16 - desni ručni zglob
#     hip_centers = []       # sredina levog (23) i desnog (24) kuka
#
#     while True:
#         success, frame = cap.read()
#
#         if not success:
#             break
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb
#         )
#
#         timestamp_ms = int((frame_count / fps) * 1000)
#
#         result = detector.detect_for_video(
#             mp_image,
#             timestamp_ms
#         )
#
#         if result.pose_landmarks:
#             lm = result.pose_landmarks[0]
#
#             left_positions.append(
#                 (lm[15].x, lm[15].y)
#             )
#
#             right_positions.append(
#                 (lm[16].x, lm[16].y)
#             )
#
#             hip_cx = (lm[23].x + lm[24].x) / 2
#             hip_cy = (lm[23].y + lm[24].y) / 2
#
#             hip_centers.append(
#                 (hip_cx, hip_cy)
#             )
#
#         else:
#             left_positions.append(None)
#             right_positions.append(None)
#             hip_centers.append(None)
#
#         frame_count += 1
#
#     cap.release()
#     detector.close()
#
#     # -----------------------------------
#     # Brzine kretanja ručnih zglobova
#     # -----------------------------------
#
#     def compute_speeds(positions):
#         speeds = []
#
#         for i in range(1, len(positions)):
#
#             if positions[i] is not None and positions[i - 1] is not None:
#
#                 dx = positions[i][0] - positions[i - 1][0]
#                 dy = positions[i][1] - positions[i - 1][1]
#
#                 speed = np.sqrt(dx ** 2 + dy ** 2)
#
#                 speeds.append(speed)
#
#         return speeds
#
#     # -----------------------------------
#     # Rastojanje ručnog zgloba od kukova
#     # -----------------------------------
#
#     def compute_distances_from_hip(positions, centers):
#
#         distances = []
#
#         for pos, center in zip(positions, centers):
#
#             if pos is not None and center is not None:
#
#                 dx = pos[0] - center[0]
#                 dy = pos[1] - center[1]
#
#                 distance = np.sqrt(dx ** 2 + dy ** 2)
#
#                 distances.append(distance)
#
#         return distances
#
#     left_speeds = compute_speeds(left_positions)
#     right_speeds = compute_speeds(right_positions)
#
#     left_dist = compute_distances_from_hip(
#         left_positions,
#         hip_centers
#     )
#
#     right_dist = compute_distances_from_hip(
#         right_positions,
#         hip_centers
#     )
#
#     # Ako nema dovoljno detektovanih tačaka
#     if not left_speeds or not right_speeds or not left_dist or not right_dist:
#         print(f"{os.path.basename(video_path)}: nema dovoljno podataka")
#         return
#
#     # -----------------------------------
#     # Srednja brzina
#     # -----------------------------------
#
#     left_speed_mean = np.mean(left_speeds)
#     right_speed_mean = np.mean(right_speeds)
#
#     # -----------------------------------
#     # Opseg pokreta
#     # -----------------------------------
#
#     left_range = np.max(left_dist) - np.min(left_dist)
#     right_range = np.max(right_dist) - np.min(right_dist)
#
#     print(f"{os.path.basename(video_path)}:")
#     print(
#         f"  Brzina - levi: {left_speed_mean:.4f}   "
#         f"desni: {right_speed_mean:.4f}"
#     )
#
#     print(
#         f"  Opseg pokreta - levi: {left_range:.4f}   "
#         f"desni: {right_range:.4f}"
#     )
#
#     # -----------------------------------
#     # Normalizacija rezultata
#     # -----------------------------------
#
#     speed_total = left_speed_mean + right_speed_mean
#     range_total = left_range + right_range
#
#     left_speed_norm = (
#         left_speed_mean / speed_total
#         if speed_total > 0 else 0.5
#     )
#
#     right_speed_norm = (
#         right_speed_mean / speed_total
#         if speed_total > 0 else 0.5
#     )
#
#     left_range_norm = (
#         left_range / range_total
#         if range_total > 0 else 0.5
#     )
#
#     right_range_norm = (
#         right_range / range_total
#         if range_total > 0 else 0.5
#     )
#
#     # -----------------------------------
#     # Glasanje
#     # -----------------------------------
#
#     left_score = left_speed_norm + left_range_norm
#     right_score = right_speed_norm + right_range_norm
#
#     speed_vote = (
#         "DESNA"
#         if right_speed_mean > left_speed_mean
#         else "LEVA"
#     )
#
#     range_vote = (
#         "DESNA"
#         if right_range > left_range
#         else "LEVA"
#     )
#
#     combined = (
#         "DESNA"
#         if right_score > left_score
#         else "LEVA"
#     )
#
#     print(
#         f"  Glas po brzini: {speed_vote}   "
#         f"Glas po opsegu: {range_vote}"
#     )
#
#     print(
#         f"  --> Kombinovana predikcija dominantne ruke: "
#         f"{combined}"
#     )
#
#     print()
#
#
# # ==========================================
# # ANALIZA CELOG FOLDERA
# # ==========================================
#
# folder_path = "dataset/igrac1/forehand"
#
# video_extensions = (
#     ".mp4",
#     ".avi",
#     ".mov",
#     ".mkv"
# )
#
# detector = make_detector()
#
# for filename in sorted(os.listdir(folder_path)):
#
#     if filename.lower().endswith(video_extensions):
#
#         video_path = os.path.join(
#             folder_path,
#             filename
#         )
#
#         analyze_dominant_hand(video_path)
#
# detector.close()

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"


def make_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.PoseLandmarker.create_from_options(options)


def analyze_dominant_hand(video_path):

    detector = make_detector()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0

    left_relative_x = []
    right_relative_x = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((frame_count / fps) * 1000)

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            lm = result.pose_landmarks[0]

            # Ramena
            left_shoulder = lm[11]
            right_shoulder = lm[12]

            # Ručni zglobovi
            left_wrist = lm[15]
            right_wrist = lm[16]

            # Centar između ramena
            shoulder_center_x = (
                left_shoulder.x + right_shoulder.x
            ) / 2

            # Relativni horizontalni položaj zgloba
            left_relative_x.append(
                left_wrist.x - shoulder_center_x
            )

            right_relative_x.append(
                right_wrist.x - shoulder_center_x
            )

        frame_count += 1

    cap.release()
    detector.close()

    if not left_relative_x or not right_relative_x:
        print(f"{os.path.basename(video_path)}: nema dovoljno podataka")
        return

    # Opseg relativnog položaja kroz video
    left_range = (
        np.max(left_relative_x)
        - np.min(left_relative_x)
    )

    right_range = (
        np.max(right_relative_x)
        - np.min(right_relative_x)
    )

    print(f"{os.path.basename(video_path)}:")
    print(
        f"  Relativni opseg - leva ruka: {left_range:.4f}   "
        f"desna ruka: {right_range:.4f}"
    )

    # Glasanje
    if right_range > left_range:
        prediction = "DESNA"
    else:
        prediction = "LEVA"

    print(
        f"  --> Predikcija dominantne ruke: {prediction}"
    )

    print()


# ==========================================
# ANALIZA CELOG FOLDERA
# ==========================================

folder_path = "dataset/igrac8_left/forehand"

video_extensions = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)

for filename in sorted(os.listdir(folder_path)):

    if filename.lower().endswith(video_extensions):

        video_path = os.path.join(
            folder_path,
            filename
        )

        analyze_dominant_hand(video_path)