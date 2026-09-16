from insightface.app import FaceAnalysis


class FaceEncoder:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, face_image):

        embedding_model = self.app.models["recognition"]

        embedding = embedding_model.get_feat(face_image)

        return embedding.flatten()