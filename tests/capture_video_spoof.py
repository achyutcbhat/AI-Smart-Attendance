import cv2
import os

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector


SAVE_FOLDER = "data/liveness/spoof/video_replay"
TOTAL_IMAGES = 20

os.makedirs(SAVE_FOLDER, exist_ok=True)

camera = CameraManager()
detector = FaceDetector()

camera.start()

count = 0

try:
    while count < TOTAL_IMAGES:

        frame = camera.read_frame()

        faces = detector.detect_faces(frame)

        if len(faces) == 1:

            x1, y1, x2, y2 = faces[0]["box"]

            face = frame[y1:y2, x1:x2]

            if face.size > 0:

                filename = os.path.join(
                    SAVE_FOLDER,
                    f"video_{count + 1:03d}.jpg"
                )

                cv2.imwrite(filename, face)

                count += 1

                print(f"Saved image {count}/{TOTAL_IMAGES}")

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            frame,
            f"Video Spoof: {count}/{TOTAL_IMAGES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Video Replay Spoof Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()

print("Video-replay spoof capture completed.")