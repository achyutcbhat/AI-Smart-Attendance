import cv2
import numpy as np


class FaceAligner:

    def __init__(self, output_size=(112, 112)):

        self.output_size = output_size

        self.reference_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def align(self, face, landmarks):

        source_points = np.array([
            landmarks["left_eye"],
            landmarks["right_eye"],
            landmarks["nose"],
            landmarks["mouth_left"],
            landmarks["mouth_right"]
        ], dtype=np.float32)

        # Convert full-frame landmark coordinates
        # to cropped-face coordinates
        x_offset = landmarks.get("_x_offset", 0)
        y_offset = landmarks.get("_y_offset", 0)

        source_points[:, 0] -= x_offset
        source_points[:, 1] -= y_offset

        transform_matrix, _ = cv2.estimateAffinePartial2D(
            source_points,
            self.reference_points
        )

        if transform_matrix is None:
            return None

        aligned_face = cv2.warpAffine(
            face,
            transform_matrix,
            self.output_size
        )

        return aligned_face