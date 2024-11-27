import cv2
import numpy as np


def apply_average_blur(image, kernel_size):

    blurred = cv2.blur(image, (kernel_size, kernel_size))
    return blurred


def apply_gaussian_blur(image, kernel_size, sigma):

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred




