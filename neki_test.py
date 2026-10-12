import cv2
import os

folder = "dataset/random_test/forehand"  # zameni pravom putanjom
for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    path = os.path.join(folder, filename)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    opened = cap.isOpened()
    print(f"{filename}: opened={opened}, frames={n}")
    cap.release()