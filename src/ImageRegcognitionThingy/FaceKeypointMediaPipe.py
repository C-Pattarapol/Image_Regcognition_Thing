# face_keypoints_mediapipe.py
import cv2, mediapipe as mp, numpy as np

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
DRAW_SPEC = mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
with mp_face.FaceMesh(static_image_mode=False,
                      max_num_faces=1,
                      refine_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as fm:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face, mp_face.FACEMESH_TESSELATION, DRAW_SPEC, DRAW_SPEC)
                # convert normalized LM to pixel coordinates
                pts = np.array([[p.x * w, p.y * h] for p in face.landmark])
        cv2.imshow('Face Keypoints', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.getWindowProperty('Face Keypoints', cv2.WND_PROP_VISIBLE) < 1:
            break


cap.release()
cv2.destroyAllWindows()