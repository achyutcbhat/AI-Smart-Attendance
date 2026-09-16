from src.liveness.background_check.feature_extractor import (
    BackgroundFeatureExtractor
)

from src.liveness.background_check.background_model import (
    BackgroundModel
)


class BackgroundAnalyzer:

    def __init__(self):

        self.extractor = BackgroundFeatureExtractor()
        self.model = BackgroundModel()

    def analyze(self, frame):

        features = self.extractor.extract(frame)

        score = self.model.score(features)

        return {
            "background_score": score,
            "acceptable": score >= 0.5
        }