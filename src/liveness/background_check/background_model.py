import numpy as np


class BackgroundModel:

    def __init__(self):
        self.expected_features = None

    def fit(self, features):

        self.expected_features = np.mean(
            features,
            axis=0
        )

    def score(self, features):

        if self.expected_features is None:
            return 0.0

        distance = np.linalg.norm(
            features - self.expected_features
        )

        score = 1.0 / (1.0 + distance)

        return float(score)