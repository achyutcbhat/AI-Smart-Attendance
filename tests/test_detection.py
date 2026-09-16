import cv2

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector


camera = CameraManager()
detector = FaceDetector()

try:
    camera.start()

    while True:
        frame = camera.read_frame()

        faces = detector.detect_faces(frame)

        for face in faces:
            x1, y1, x2, y2 = face["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()