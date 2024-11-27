import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise


# add Gaussian noise
def add_gaussian_noise(img):
    # adaugam zgomot Gaussian cu media 0 și deviatia standard 25
    gauss = np.random.normal(0, 25, img.shape).astype('float32')

    noisy_img = cv2.add(img.astype(np.float32), gauss)

    # limitezz valorile la intervalul [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')

    return noisy_img


# add Salt-and-Pepper noise
def add_salt_and_pepper_noise(img):
    noise_img = random_noise(img, mode='s&p', amount=0.2)
    noise_img = np.array(255 * noise_img, dtype='uint8')

    return noise_img


# add Speckle noise
def add_speckle_noise(img):
    gauss = np.random.normal(0, 1, img.shape)
    noisy_img = img + img * gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')
    return noisy_img
