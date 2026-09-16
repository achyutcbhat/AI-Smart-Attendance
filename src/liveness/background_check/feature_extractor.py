import cv2
import numpy as np


class BackgroundFeatureExtractor:

    def extract(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.mean(edges > 0)

        return np.array([
            mean_brightness,
            brightness_std,
            edge_ratio
        ], dtype=np.float32)