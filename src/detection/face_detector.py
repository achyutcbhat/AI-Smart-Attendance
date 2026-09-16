from retinaface import RetinaFace


class FaceDetector:

    def __init__(self):
        self.detector = RetinaFace

    def detect_faces(self, frame):

        detections = self.detector.detect_faces(frame)

        faces = []

        if not detections:
            return faces

        for _, data in detections.items():

            facial_area = data["facial_area"]
            landmarks = data.get("landmarks", {})

            x1, y1, x2, y2 = facial_area

            landmarks["_x_offset"] = x1
            landmarks["_y_offset"] = y1

            faces.append({
                "box": (x1, y1, x2, y2),
                "landmarks": landmarks
            })

        return faces