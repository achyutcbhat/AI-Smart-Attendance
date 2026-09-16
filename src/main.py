import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector
from src.detection.face_alignment import FaceAligner
from src.recognition.face_encoder import FaceEncoder
from src.recognition.identity_matcher import IdentityMatcher
from src.liveness.liveness_checker import LivenessChecker
from src.liveness.background_check.background_analyzer import BackgroundAnalyzer
from src.liveness.fusion.score_fusion import ScoreFusion
from src.attendance.attendance_manager import AttendanceManager
from src.database.database import get_connection
from src.utils.logger import get_logger


logger = get_logger("smart_attendance")


def load_registered_embeddings():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT student_id, embedding_path FROM students"
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    embeddings = {}

    for student_id, embedding_path in rows:
        embeddings[student_id] = np.load(embedding_path)

    return embeddings


def main():

    camera = CameraManager()
    detector = FaceDetector()
    aligner = FaceAligner()
    encoder = FaceEncoder()
    matcher = IdentityMatcher()
    liveness = LivenessChecker()
    background = BackgroundAnalyzer()
    fusion = ScoreFusion()
    attendance = AttendanceManager()

    registered_embeddings = load_registered_embeddings()

    if not registered_embeddings:

        logger.warning("No registered students found.")

        print("No registered students found.")
        return

    camera.start()

    logger.info("Smart Attendance System Started")

    print("Smart Attendance System Started")
    print("Press Q to quit.")

    try:

        while True:

            frame = camera.read_frame()

            faces = detector.detect_faces(frame)

            if len(faces) == 1:

                face = faces[0]

                x1, y1, x2, y2 = face["box"]

                cropped_face = frame[y1:y2, x1:x2]

                aligned_face = aligner.align(
                    cropped_face,
                    face["landmarks"]
                )

                if aligned_face is not None:

                    embedding = encoder.get_embedding(
                        aligned_face
                    )

                    student_id, recognition_score = matcher.match(
                        embedding,
                        registered_embeddings
                    )

                    # Unknown face
                    if student_id is None:

                        logger.warning(
                            f"Unknown face detected. "
                            f"Recognition score: {recognition_score:.2f}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Unknown Face",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                    else:

                        # Liveness check
                        live_result = liveness.check(
                            aligned_face
                        )

                        # Background check
                        background_result = background.analyze(
                            frame
                        )

                        # Score fusion
                        final_score = fusion.calculate(
                            recognition_score,
                            live_result["score"],
                            background_result["background_score"]
                        )

                        # Attendance decision
                        result = attendance.mark_attendance(
                            student_id,
                            recognition_score,
                            live_result["score"],
                            background_result["background_score"],
                            final_score
                        )

                        logger.info(
                            f"Student ID: {student_id}, "
                            f"Recognition: {recognition_score:.2f}, "
                            f"Liveness: {live_result['score']:.2f}, "
                            f"Background: {background_result['background_score']:.2f}, "
                            f"Final: {final_score:.2f}, "
                            f"Result: {result}"
                        )

                        # Display face box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        # Student ID
                        cv2.putText(
                            frame,
                            f"ID: {student_id}",
                            (x1, y1 - 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # Recognition score
                        cv2.putText(
                            frame,
                            f"Recognition: {recognition_score:.2f}",
                            (x1, y1 - 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        # Liveness score
                        cv2.putText(
                            frame,
                            f"Liveness: {live_result['score']:.2f}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        # Final decision
                        cv2.putText(
                            frame,
                            f"{result} | Final: {final_score:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

            elif len(faces) > 1:

                logger.warning("Multiple faces detected.")

                cv2.putText(
                    frame,
                    "Multiple Faces Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "AI Smart Attendance",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Smart Attendance System stopped.")
                break

    finally:

        camera.release()


if __name__ == "__main__":
    main()