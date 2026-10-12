# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import os
#
# MODEL_PATH = "pose_landmarker_full.task"
#
# # ------------------------------------------------
# # PODESAVANJA
# # ------------------------------------------------
#
# # Folderi sa igrac3 uglovnim snimcima koje treba zumirati.
# # Ocekivana struktura: dataset/<folder>/backhand/*.mp4, dataset/<folder>/forehand/*.mp4
# SOURCE_FOLDERS = [
#     "dataset/igrac3_angle_fh"
# ]
#
# # Koliko dodatnog prostora (kao % dimenzije bounding box-a) ostaviti oko igraca
# PADDING_FRACTION = 0.20
#
# # Faktor uvecanja izlaznog videa (2.0 = duplo vece dimenzije crop-a)
# UPSCALE_FACTOR = 2.0
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
# def process_video(input_path, output_path):
#     detector = make_detector()
#     cap = cv2.VideoCapture(input_path)
#
#     if not cap.isOpened():
#         print(f"    Ne mogu da otvorim: {input_path}")
#         detector.close()
#         return False
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # ------------------------------------------------
#     # PRVI PROLAZ: ucitaj sve frejmove i nadji granice tela
#     # ------------------------------------------------
#
#     frames = []
#     all_x = []
#     all_y = []
#
#     frame_count = 0
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         frames.append(frame)
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         timestamp_ms = int((frame_count / fps) * 1000)
#         result = detector.detect_for_video(mp_image, timestamp_ms)
#
#         if result.pose_landmarks:
#             for lm in result.pose_landmarks[0]:
#                 if lm.visibility > 0.3:  # ignorisi jako nepouzdane tacke
#                     all_x.append(lm.x)
#                     all_y.append(lm.y)
#
#         frame_count += 1
#
#     cap.release()
#     detector.close()
#
#     if len(frames) == 0 or len(all_x) == 0:
#         print(f"    Nije detektovana nijedna pouzdana poza, preskacem: {input_path}")
#         return False
#
#     # ------------------------------------------------
#     # IZRACUNAJ BOUNDING BOX (u pikselima) + padding
#     # ------------------------------------------------
#
#     min_x, max_x = min(all_x), max(all_x)
#     min_y, max_y = min(all_y), max(all_y)
#
#     box_w = max_x - min_x
#     box_h = max_y - min_y
#
#     pad_x = box_w * PADDING_FRACTION
#     pad_y = box_h * PADDING_FRACTION
#
#     min_x = max(0.0, min_x - pad_x)
#     max_x = min(1.0, max_x + pad_x)
#     min_y = max(0.0, min_y - pad_y)
#     max_y = min(1.0, max_y + pad_y)
#
#     px_x1 = int(min_x * frame_w)
#     px_x2 = int(max_x * frame_w)
#     px_y1 = int(min_y * frame_h)
#     px_y2 = int(max_y * frame_h)
#
#     crop_w = max(2, px_x2 - px_x1)
#     crop_h = max(2, px_y2 - px_y1)
#
#     out_w = int(crop_w * UPSCALE_FACTOR)
#     out_h = int(crop_h * UPSCALE_FACTOR)
#     # dimenzije moraju biti parne za neke kodeke
#     out_w += out_w % 2
#     out_h += out_h % 2
#
#     # ------------------------------------------------
#     # DRUGI "PROLAZ": iseci i uvecaj svaki frejm, upisi video
#     # ------------------------------------------------
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
#
#     for frame in frames:
#         cropped = frame[px_y1:px_y2, px_x1:px_x2]
#         resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
#         writer.write(resized)
#
#     writer.release()
#
#     print(f"    OK -> {output_path}  (crop {crop_w}x{crop_h} -> {out_w}x{out_h})")
#     return True
#
#
# # ------------------------------------------------
# # GLAVNA PETLJA - PROLAZAK KROZ SVE FOLDERE I FAJLOVE
# # ------------------------------------------------
#
# total_ok = 0
# total_failed = 0
#
# for source_folder in SOURCE_FOLDERS:
#
#     if not os.path.isdir(source_folder):
#         print(f"UPOZORENJE: folder ne postoji: {source_folder}")
#         continue
#
#     # ime izlaznog foldera: isto ime + "_zoomed"
#     dest_folder = source_folder + "_zoomed"
#
#     print(f"\n=== Obrada: {source_folder} -> {dest_folder} ===")
#
#     for stroke_name in ["backhand", "forehand"]:
#         stroke_path = os.path.join(source_folder, stroke_name)
#
#         if not os.path.isdir(stroke_path):
#             continue
#
#         for filename in sorted(os.listdir(stroke_path)):
#             if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
#                 continue
#
#             input_path = os.path.join(stroke_path, filename)
#             output_path = os.path.join(dest_folder, stroke_name, filename)
#
#             print(f"  Obrada: {stroke_name}/{filename}")
#
#             success = process_video(input_path, output_path)
#             if success:
#                 total_ok += 1
#             else:
#                 total_failed += 1
#
# print()
# print("===================================")
# print("GOTOVO")
# print("===================================")
# print(f"Uspesno obradjeno: {total_ok}")
# print(f"Neuspesno: {total_failed}")
# print()
# print("Novi folderi (dodaj ih u dataset/ ako vec nisu tu, pa pokreni")
# print("build_dataset_add_player.py sa odgovarajucim imenima da ih dodas u X.npy/y.npy/players.npy):")
# for source_folder in SOURCE_FOLDERS:
#     print(f"  {source_folder}_zoomed")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

MODEL_PATH = "pose_landmarker_full.task"

# ------------------------------------------------
# PODESAVANJA
# ------------------------------------------------

# Folderi sa igrac3 uglovnim snimcima koje treba zumirati.
# Ocekivana struktura: dataset/<folder>/backhand/*.mp4, dataset/<folder>/forehand/*.mp4
SOURCE_FOLDERS = [
    "dataset/zoom_these"
]

# Koliko dodatnog prostora (kao % dimenzije bounding box-a) ostaviti oko igraca
PADDING_FRACTION = 0.20

# Faktor uvecanja izlaznog videa
UPSCALE_FACTOR = 2.0

# Donjih 35% originalnog kadra se izbacuje zbog senke.
# Crop se zavrsava na 65% visine kadra.
BOTTOM_CROP_FRACTION = 0.30


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


def process_video(input_path, output_path):
    detector = make_detector()
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"    Ne mogu da otvorim: {input_path}")
        detector.close()
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------
    # PRVI PROLAZ: ucitaj sve frejmove i nadji granice tela
    # ------------------------------------------------

    frames = []
    all_x = []
    all_y = []

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

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
            for lm in result.pose_landmarks[0]:
                if lm.visibility > 0.3:
                    all_x.append(lm.x)
                    all_y.append(lm.y)

        frame_count += 1

    cap.release()
    detector.close()

    if len(frames) == 0 or len(all_x) == 0:
        print(
            f"    Nije detektovana nijedna pouzdana poza, "
            f"preskacem: {input_path}"
        )
        return False

    # ------------------------------------------------
    # IZRACUNAJ BOUNDING BOX + PADDING
    # ------------------------------------------------

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    box_w = max_x - min_x
    box_h = max_y - min_y

    pad_x = box_w * PADDING_FRACTION
    pad_y = box_h * PADDING_FRACTION

    min_x = max(0.0, min_x - pad_x)
    max_x = min(1.0, max_x + pad_x)

    min_y = max(0.0, min_y - pad_y)
    max_y = min(1.0, max_y + pad_y)

    # ------------------------------------------------
    # HORIZONTALNE GRANICE
    # ------------------------------------------------

    px_x1 = int(min_x * frame_w)
    px_x2 = int(max_x * frame_w)

    # ------------------------------------------------
    # VERTIKALNE GRANICE
    # ------------------------------------------------

    # Gornja granica i dalje zavisi od igraca
    px_y1 = int(min_y * frame_h)

    # Donjih 35% originalnog kadra se izbacuje.
    # Crop se zavrsava na 65% visine slike.
    px_y2 = int(frame_h * (1.0 - BOTTOM_CROP_FRACTION))

    # Osiguraj da je donja granica dovoljno velika
    px_y2 = max(px_y2, px_y1 + 2)

    crop_w = max(2, px_x2 - px_x1)
    crop_h = max(2, px_y2 - px_y1)

    # ------------------------------------------------
    # UVECAJ CROP
    # ------------------------------------------------

    out_w = int(crop_w * UPSCALE_FACTOR)
    out_h = int(crop_h * UPSCALE_FACTOR)

    # dimenzije moraju biti parne za neke kodeke
    out_w += out_w % 2
    out_h += out_h % 2

    # ------------------------------------------------
    # DRUGI PROLAZ:
    # ISECI I UVECAJ SVAKI FREJM
    # ------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (out_w, out_h)
    )

    for frame in frames:

        cropped = frame[
            px_y1:px_y2,
            px_x1:px_x2
        ]

        resized = cv2.resize(
            cropped,
            (out_w, out_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        writer.write(resized)

    writer.release()

    print(
        f"    OK -> {output_path} "
        f"(crop {crop_w}x{crop_h} -> {out_w}x{out_h})"
    )

    return True


# ------------------------------------------------
# GLAVNA PETLJA
# ------------------------------------------------

total_ok = 0
total_failed = 0

for source_folder in SOURCE_FOLDERS:

    if not os.path.isdir(source_folder):
        print(
            f"UPOZORENJE: folder ne postoji: "
            f"{source_folder}"
        )
        continue

    # ime izlaznog foldera:
    # isto ime + "_zoomed"
    dest_folder = source_folder + "_zoomed_new"

    print(
        f"\n=== Obrada: {source_folder} -> "
        f"{dest_folder} ==="
    )

    for stroke_name in ["backhand", "forehand"]:

        stroke_path = os.path.join(
            source_folder,
            stroke_name
        )

        if not os.path.isdir(stroke_path):
            continue

        for filename in sorted(os.listdir(stroke_path)):

            if not filename.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                continue

            input_path = os.path.join(
                stroke_path,
                filename
            )

            output_path = os.path.join(
                dest_folder,
                stroke_name,
                filename
            )

            print(
                f"  Obrada: "
                f"{stroke_name}/{filename}"
            )

            success = process_video(
                input_path,
                output_path
            )

            if success:
                total_ok += 1
            else:
                total_failed += 1


print()
print("===================================")
print("GOTOVO")
print("===================================")
print(f"Uspesno obradjeno: {total_ok}")
print(f"Neuspesno: {total_failed}")
print()

print(
    "Novi folderi (dodaj ih u dataset/ "
    "ako vec nisu tu, pa pokreni"
)

print(
    "build_dataset_add_player.py sa "
    "odgovarajucim imenima da ih dodas "
    "u X.npy/y.npy/players.npy):"
)

for source_folder in SOURCE_FOLDERS:
    print(f"  {source_folder}_zoomed_new")

