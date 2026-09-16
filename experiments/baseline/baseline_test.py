import cv2

from src.recognition.face_encoder import FaceEncoder
from src.recognition.identity_matcher import IdentityMatcher


image = cv2.imread(
    "data/liveness/real/real_001.jpg"
)

encoder = FaceEncoder()
matcher = IdentityMatcher()


embedding = encoder.get_embedding(image)

print("Baseline Recognition Test")
print("Embedding shape:", embedding.shape)

print("Recognition system working.")