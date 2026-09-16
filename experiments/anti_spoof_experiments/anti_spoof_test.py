import cv2

from src.liveness.liveness_checker import LivenessChecker


checker = LivenessChecker()

images = {
    "Real": "data/liveness/real/real_001.jpg",
    "Mobile Spoof": "data/liveness/spoof/mobile_001.jpg",
    "Video Spoof": "data/liveness/spoof/video_003.jpg"
}

for name, path in images.items():

    image = cv2.imread(path)

    result = checker.check(image)

    print(f"\n{name}")
    print("Label:", result["label"])
    print("Liveness Score:", result["score"])