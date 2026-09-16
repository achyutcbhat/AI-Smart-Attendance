import cv2

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector
from src.detection.face_alignment import FaceAligner


camera = CameraManager()
detector = FaceDetector()
aligner = FaceAligner()

try:
    camera.start()

    while True:
        frame = camera.read_frame()

        faces = detector.detect_faces(frame)

        for face_data in faces:
            x1, y1, x2, y2 = face_data["box"]
            landmarks = face_data["landmarks"]

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            aligned_face = aligner.align(face, landmarks)

            if aligned_face is not None:
                cv2.imshow("Aligned Face", aligned_face)

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