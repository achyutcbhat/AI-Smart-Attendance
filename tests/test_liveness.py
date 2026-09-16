import cv2

from src.liveness.liveness_checker import LivenessChecker


checker = LivenessChecker()

# Test real image
real_image = cv2.imread("data/liveness/real/real_001.jpg")
real_result = checker.check(real_image)

print("\nREAL IMAGE")
print(real_result)


# Test mobile-photo spoof
mobile_image = cv2.imread(
    "data/liveness/spoof/mobile_001.jpg"
)
mobile_result = checker.check(mobile_image)

print("\nMOBILE PHOTO SPOOF")
print(mobile_result)


# Test video-replay spoof
video_image = cv2.imread(
    "data/liveness/spoof/video_001.jpg"
)
video_result = checker.check(video_image)

print("\nVIDEO REPLAY SPOOF")
print(video_result)