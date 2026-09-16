import numpy as np

from src.utils.config import load_config


class IdentityMatcher:

    def __init__(self):
        config = load_config()
        self.threshold = config["recognition"]["threshold"]

    def cosine_similarity(self, embedding1, embedding2):
        embedding1 = np.asarray(embedding1)
        embedding2 = np.asarray(embedding2)

        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) *
            np.linalg.norm(embedding2)
        )

    def match(self, live_embedding, registered_embeddings):
        best_student = None
        best_score = -1

        for student_id, registered_embedding in registered_embeddings.items():

            score = self.cosine_similarity(
                live_embedding,
                registered_embedding
            )

            if score > best_score:
                best_score = score
                best_student = student_id

        if best_score >= self.threshold:
            return best_student, best_score

        return None, best_score