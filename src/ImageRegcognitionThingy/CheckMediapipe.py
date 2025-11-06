# save as check_mediapipe.py
import cv2, mediapipe as mp
cap = cv2.VideoCapture(0)
with mp.solutions.face_mesh.FaceMesh() as fm:
    ret, frame = cap.read()
    if not ret:
        raise SystemExit('No camera frame')
    print('Camera OK, mediapipe ready')
cap.release()