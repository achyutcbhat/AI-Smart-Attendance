import torch

from src.liveness.anti_spoof.model import AntiSpoofCNN
from src.liveness.anti_spoof.preprocessing import preprocess_face


class AntiSpoofPredictor:

    def __init__(self, model_path=None):

        self.model = AntiSpoofCNN()

        if model_path:
            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location="cpu"
                )
            )

        self.model.eval()

    def predict(self, face_image):

        face_tensor = preprocess_face(face_image)

        with torch.no_grad():
            output = self.model(face_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

        real_score = probabilities[0][0].item()
        spoof_score = probabilities[0][1].item()

        if real_score >= spoof_score:
            label = "real"
        else:
            label = "spoof"

        return {
            "label": label,
            "real_score": real_score,
            "spoof_score": spoof_score
        }