from src.liveness.anti_spoof.predictor import AntiSpoofPredictor
from src.utils.config import load_config


class LivenessChecker:

    def __init__(
        self,
        model_path="models/anti_spoof/anti_spoof_cnn.pth"
    ):
        config = load_config()

        self.threshold = config["liveness"]["threshold"]

        self.predictor = AntiSpoofPredictor(model_path)

    def check(self, face_image):

        result = self.predictor.predict(face_image)

        is_live = (
            result["label"] == "real"
            and result["real_score"] >= self.threshold
        )

        return {
            "is_live": is_live,
            "score": result["real_score"],
            "label": result["label"]
        }