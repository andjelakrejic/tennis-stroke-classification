import cv2
cap = cv2.VideoCapture("dataset/igrac3_45_bh/backhand/bh_05.MOV")  # ili putanja do bilo kog novog 4K klipa
ret, frame = cap.read()
print("Shape:", frame.shape)
cv2.imwrite("test_frame.jpg", frame)
cap.release()