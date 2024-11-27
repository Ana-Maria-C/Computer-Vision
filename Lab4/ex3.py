import cv2
import numpy as np


def rotate_image(img, angle):

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated


def shear_image(img, shear_factor_x=0.0, shear_factor_y=0.0):

    (h, w) = img.shape[:2]

    shear_matrix = np.array([
        [1, shear_factor_x, 0],
        [shear_factor_y, 1, 0]
    ], dtype=np.float32)

    sheared = cv2.warpAffine(img, shear_matrix, (w, h))
    return sheared



