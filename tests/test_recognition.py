import cv2

from src.camera.camera_manager import CameraManager
from src.recognition.face_encoder import FaceEncoder


camera = CameraManager()
encoder = FaceEncoder()

try:
    camera.start()

    while True:
        frame = camera.read_frame()

        embedding = encoder.get_embedding(frame)

        if embedding is not None:
            print("Face embedding generated:", embedding.shape)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()