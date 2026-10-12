import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

VIDEO_PATH = "dataset/video8.mp4"
OUTPUT_CLIPS_DIR = "unused//rally_saska"

WRIST_LANDMARK = 16
ELBOW_LANDMARK = 14

os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

wrist_positions = []
elbow_positions = []

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

        landmarks = result.pose_landmarks[0]

        wrist = landmarks[WRIST_LANDMARK]
        elbow = landmarks[ELBOW_LANDMARK]

        wrist_positions.append(
            (wrist.x, wrist.y)
        )

        elbow_positions.append(
            (elbow.x, elbow.y)
        )

    else:

        wrist_positions.append(
            wrist_positions[-1]
            if wrist_positions
            else (0, 0)
        )

        elbow_positions.append(
            elbow_positions[-1]
            if elbow_positions
            else (0, 0)
        )

    frame_count += 1


cap.release()
detector.close()

print(
    f"Obradjeno {frame_count} frejmova."
)


# ------------------------------------------------
# Brzina zgloba i lakta
# ------------------------------------------------

wrist_positions = np.array(
    wrist_positions
)

elbow_positions = np.array(
    elbow_positions
)


wrist_velocity = np.linalg.norm(
    np.diff(
        wrist_positions,
        axis=0
    ),
    axis=1
)


elbow_velocity = np.linalg.norm(
    np.diff(
        elbow_positions,
        axis=0
    ),
    axis=1
)


# ------------------------------------------------
# Kombinovani signal
# ------------------------------------------------

velocity = (
    0.7 * wrist_velocity +
    0.3 * elbow_velocity
)


# ------------------------------------------------
# Adaptivni threshold
# ------------------------------------------------

threshold = (
    np.mean(velocity) +
    1.5 * np.std(velocity)
)

print(
    f"Adaptivni threshold: "
    f"{threshold:.4f}"
)


raw_peaks, properties = find_peaks(
    velocity,
    height=threshold,
    distance=int(fps * 0.5)
)


print(
    "\nSirovi kandidati "
    "(pre grupisanja):"
)

for p in raw_peaks:

    print(
        f"  Frame {p} "
        f"({p / fps:.2f}s): "
        f"signal = {velocity[p]:.4f}"
    )


# ------------------------------------------------
# Grupisanje bliskih peakova
# ------------------------------------------------

def merge_nearby_peaks(
        peaks,
        velocity,
        min_gap_frames
):

    if len(peaks) == 0:
        return []

    groups = [[peaks[0]]]

    for p in peaks[1:]:

        first_peak_in_group = (
            groups[-1][0]
        )

        if (
            p - first_peak_in_group
            <= min_gap_frames
        ):

            groups[-1].append(p)

        else:

            groups.append([p])


    final_peaks = [

        max(
            group,
            key=lambda idx:
            velocity[idx]
        )

        for group in groups
    ]

    return final_peaks


final_peaks = merge_nearby_peaks(
    raw_peaks,
    velocity,
    min_gap_frames=int(
        fps * 1.5
    )
)


print(
    "\nFinalni udarci "
    "(posle grupisanja):"
)

for p in final_peaks:

    print(
        f"  Frame {p} "
        f"({p / fps:.2f}s): "
        f"signal = {velocity[p]:.4f}"
    )


print(
    "\nBroj detektovanih udaraca:",
    len(final_peaks)
)


# ------------------------------------------------
# Graf
# ------------------------------------------------

plt.figure(
    figsize=(14, 5)
)


# Zglob posebno
plt.plot(
    wrist_velocity,
    label="Брзина зглоба",
    alpha=0.6
)


# Lakat posebno
plt.plot(
    elbow_velocity,
    label="Брзина лакта",
    alpha=0.6
)


# Kombinovani signal
plt.plot(
    velocity,
    label="Комбиновани сигнал",
    linewidth=2
)


# Threshold
plt.axhline(
    threshold,
    linestyle="--",
    label="Адаптивни праг"
)


# Sirovi kandidati
plt.plot(
    raw_peaks,
    velocity[raw_peaks],
    "yx",
    markersize=8,
    label="Сирови кандидати"
)


# Finalni udarci
plt.plot(
    final_peaks,
    velocity[final_peaks],
    "ro",
    markersize=10,
    label="Финални ударац"
)


plt.xlabel(
    "Кадар"
)

plt.ylabel(
    "Брзина / комбиновани сигнал"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../combined_velocity_plot.png"
)

plt.show()

# ------------------------------------------------
# Sečenje videa
# ------------------------------------------------

PRE_BUFFER_FRAMES = int(
    fps * 0.8
)

POST_BUFFER_FRAMES = int(
    fps * 0.4
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)

frames_all = []


while True:

    success, frame = cap.read()

    if not success:
        break

    frames_all.append(frame)


cap.release()


for i, peak in enumerate(
    final_peaks
):

    start = max(
        0,
        peak - PRE_BUFFER_FRAMES
    )

    end = min(
        len(frames_all),
        peak + POST_BUFFER_FRAMES
    )

    clip_frames = (
        frames_all[start:end]
    )

    if not clip_frames:
        continue


    h, w = (
        clip_frames[0].shape[:2]
    )


    out_path = os.path.join(
        OUTPUT_CLIPS_DIR,
        f"udarac_{i:02d}.mp4"
    )


    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (w, h)
    )


    for frame in clip_frames:
        writer.write(frame)


    writer.release()


    print(
        f"Sačuvan klip: "
        f"{out_path}"
    )