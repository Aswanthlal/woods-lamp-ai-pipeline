import cv2
import numpy as np

class SkinMaskGenerator:
    def __init__(self):
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def _create_polygon(self, landmarks, indices, w, h):
        return np.array([[(landmarks[i]["x"] * w), (landmarks[i]["y"] * h)] for i in indices], np.int32).reshape((-1, 1, 2))

    def get_full_face_mask(self, image_shape, landmarks):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        face_poly = self._create_polygon(landmarks, self.FACE_OVAL, w, h)
        cv2.fillPoly(mask, [face_poly], 255)

        cv2.fillPoly(mask, [
            self._create_polygon(landmarks, self.LEFT_EYE, w, h),
            self._create_polygon(landmarks, self.RIGHT_EYE, w, h),
            self._create_polygon(landmarks, self.LIPS, w, h),
            self._create_polygon(landmarks, self.LEFT_EYEBROW, w, h),
            self._create_polygon(landmarks, self.RIGHT_EYEBROW, w, h)
        ], 0)
        return mask