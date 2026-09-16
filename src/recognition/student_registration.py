import os
import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector
from src.detection.face_alignment import FaceAligner
from src.recognition.face_encoder import FaceEncoder
from src.database.database import get_connection


class StudentRegistration:

    def __init__(self):
        self.camera = CameraManager()
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.encoder = FaceEncoder()

    def register_student(self, student_code, name, email=""):

        save_folder = os.path.join(
            "data",
            "registered_faces",
            student_code
        )

        os.makedirs(save_folder, exist_ok=True)

        print("Starting camera...")
        self.camera.start()

        embedding = None

        try:
            while True:

                frame = self.camera.read_frame()

                faces = self.detector.detect_faces(frame)

                if len(faces) == 1:

                    face = faces[0]

                    x1, y1, x2, y2 = face["box"]
                    cropped_face = frame[y1:y2, x1:x2]

                    aligned_face = self.aligner.align(
                        cropped_face,
                        face["landmarks"]
                    )

                    if aligned_face is not None:

                        embedding = self.encoder.get_embedding(
                            aligned_face
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Face detected - Press S to save",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                cv2.imshow("Student Registration", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s") and embedding is not None:
                    break

                if key == ord("q"):
                    return

        finally:
            self.camera.release()

        # Save embedding
        embedding_path = os.path.join(
            save_folder,
            "embedding.npy"
        )

        np.save(embedding_path, embedding)

        # Save student information to MySQL
        connection = get_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO students
        (student_code, name, email, embedding_path)
        VALUES (%s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                student_code,
                name,
                email,
                embedding_path
            )
        )

        connection.commit()

        cursor.close()
        connection.close()

        print("Student registered successfully.")
        print("Embedding saved at:", embedding_path)