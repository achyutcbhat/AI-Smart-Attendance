from src.utils.config import load_config


class ScoreFusion:

    def __init__(self):

        config = load_config()

        self.recognition_weight = config["fusion"]["recognition_weight"]
        self.liveness_weight = config["fusion"]["liveness_weight"]
        self.background_weight = config["fusion"]["background_weight"]

    def calculate(
        self,
        recognition_score,
        liveness_score,
        background_score
    ):

        final_score = (
            recognition_score * self.recognition_weight
            + liveness_score * self.liveness_weight
            + background_score * self.background_weight
        )

        return final_score