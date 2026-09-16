import os
import cv2

from src.liveness.anti_spoof.predictor import AntiSpoofPredictor

MODEL_PATH = "models/anti_spoof/anti_spoof_cnn.pth"

REAL_FOLDER = "data/liveness/real"
SPOOF_FOLDER = "data/liveness/spoof"


def test_folder(predictor, folder, category):
    print(f"\n===== {category} IMAGES =====")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        print(f"No images found in {folder}")
        return

    for filename in files:
        image_path = os.path.join(folder, filename)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load: {filename}")
            continue

        try:
            result = predictor.predict(image)

            print(f"{filename} -> {result}")

        except Exception as e:
            print(f"Error with {filename}: {e}")


def main():
    print("===== ANTI-SPOOF CNN TEST =====")

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    predictor = AntiSpoofPredictor(MODEL_PATH)

    test_folder(predictor, REAL_FOLDER, "REAL")
    test_folder(predictor, SPOOF_FOLDER, "SPOOF")


if __name__ == "__main__":
    main()