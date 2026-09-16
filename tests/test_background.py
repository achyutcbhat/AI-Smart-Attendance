import cv2

from src.liveness.background_check.background_analyzer import (
    BackgroundAnalyzer
)


image = cv2.imread("data/liveness/real/real_001.jpg")

analyzer = BackgroundAnalyzer()

result = analyzer.analyze(image)

print("Background result:", result)