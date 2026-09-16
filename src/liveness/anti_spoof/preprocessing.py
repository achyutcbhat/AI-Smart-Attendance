import cv2
import numpy as np
import torch


def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (224, 224))

    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    face_image = face_image.astype(np.float32) / 255.0

    face_image = np.transpose(
        face_image,
        (2, 0, 1)
    )

    face_tensor = torch.tensor(
        face_image,
        dtype=torch.float32
    ).unsqueeze(0)

    return face_tensor