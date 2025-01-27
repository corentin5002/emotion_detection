import cv2

# Test if the default camera (index 0) works
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera detected!")
else:
    print("No camera detected.")

cap.release()