import cv2
from src.camera.camera_manager import CameraManager


camera = CameraManager()

try:
    camera.start()

    while True:
        frame = camera.read_frame()

        cv2.imshow("AI Smart Attendance - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()