import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_path = "img.png"
image = cv2.imread(img_path)
image_ex_7 = cv2.imread("img_ex7.png")
img_array = np.array(image)
img_array_ex7 = np.array(image_ex_7)


def original_image():
    cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def simple_averaging():
    gray_image = (img_array[:, :, 0] / 3 + img_array[:, :, 1] / 3 + img_array[:, :, 2] / 3).astype(np.uint8)
    cv2.imshow("Grayscale (Simple Averaging)", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def weighted_average(weights, title):
    gray_image = (weights[0] * img_array[:, :, 0] +
                  weights[1] * img_array[:, :, 1] +
                  weights[2] * img_array[:, :, 2]).astype(np.uint8)
    cv2.imshow("Weighted Average:" + title, gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def desaturation():
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = int(image[i, j, 0])
            g = int(image[i, j, 1])
            b = int(image[i, j, 2])
            gray_image[i, j] = (min(r, g, b) + max(r, g, b)) // 2
    cv2.imshow("Grayscale (Desaturation)", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decomposition_max_grayscale():
    # Calculate grayscale using the maximum value
    gray_image = np.max(img_array, axis=2).astype(np.uint8)
    cv2.imshow("Decomposition (Maximum)", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decomposition_min_grayscale():
    # Calculate grayscale using the minimum value
    gray_image = np.min(img_array, axis=2).astype(np.uint8)
    cv2.imshow("Decomposition (Minimum)", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def single_color_channel_grayscale(channel):
    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0 (Red), 1 (Green), or 2 (Blue).")

    gray_image = img_array[:, :, channel].astype(np.uint8) 

    channel_names = {0: "Red", 1: "Green", 2: "Blue"}
    cv2.imshow(f"Grayscale ({channel_names[channel]})", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def custom_gray_shades(number_of_shades):
    if number_of_shades < 2 or number_of_shades > 256:
        raise ValueError("Number of shades must be between 2 and 256.")

    interval_size = 255 // number_of_shades
    grayscale_image = (img_array[:, :, 0] / 3 + img_array[:, :, 1] / 3 + img_array[:, :, 2] / 3).astype(np.uint8)
    custom_gray = (grayscale_image // interval_size) * interval_size
    cv2.imshow(f"Custom Gray Shades ({number_of_shades})", custom_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


###### custom gray with error difusion


def floyd_steinberg_dithering(num_shades):
    grayscale_image = (img_array_ex7[:, :, 0] / 3 + img_array_ex7[:, :, 1] / 3 + img_array_ex7[:, :, 2] / 3).astype(np.uint8)
    img_array_floyd = np.array(grayscale_image, dtype=np.int16)
    height, width = img_array_floyd.shape

    interval_size = 255 // num_shades

    output_image = np.zeros_like(img_array_floyd)

    for i in range(height):
        for j in range(width):
            output_image[i, j] = (img_array_floyd[i, j] // interval_size) * interval_size
            err = img_array_floyd[i, j] - output_image[i, j]
            if j < width - 1:
                img_array_floyd[i, j + 1] = np.clip(img_array_floyd[i, j + 1] + int(err * 7 // 16), 0, 255)
            if i < height - 1 and j > 0:
                img_array_floyd[i + 1, j - 1] = np.clip(img_array_floyd[i + 1, j - 1] + int(err * 3 // 16), 0, 255)
            if i < height - 1:
                img_array_floyd[i + 1, j] = np.clip(img_array_floyd[i + 1, j] + int(err * 5 // 16), 0, 255)
            if i < height - 1 and j < width - 1:
                img_array_floyd[i + 1, j + 1] = np.clip(img_array_floyd[i + 1, j + 1] + int(err // 16), 0, 255)

    output_image = output_image.astype(np.uint8)

    cv2.imshow('Floyd-Steinberg Dithered Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stucki_dithering(num_shades):
    grayscale_image = (img_array_ex7[:, :, 0] / 3 + img_array_ex7[:, :, 1] / 3 + img_array_ex7[:, :, 2] / 3).astype(
        np.uint8)
    img_array_stucki = np.array(grayscale_image, dtype=np.int16)
    height, width = img_array_stucki.shape


    output_image = np.zeros_like(img_array_stucki)
    interval_size = 255 // num_shades

    for i in range(height):
        for j in range(width):
            output_image[i, j] = (img_array_stucki[i, j] // interval_size) * interval_size
            err = img_array_stucki[i, j] - output_image[i, j]
            if j < width - 1:
                img_array_stucki[i, j + 1] = np.clip(img_array_stucki[i, j + 1] + int(err * 8 // 42), 0, 255)
            if j < width - 2:
                img_array_stucki[i, j + 2] = np.clip(img_array_stucki[i, j + 2] + int(err * 4 // 42), 0, 255)
            if i < height - 1 and j > 1:
                img_array_stucki[i + 1, j - 2] = np.clip(img_array_stucki[i + 1, j - 2] + int(err * 2 // 42), 0, 255)
            if i < height - 1 and j > 0:
                img_array_stucki[i + 1, j - 1] = np.clip(img_array_stucki[i + 1, j - 1] + int(err * 4 // 42), 0, 255)
            if i < height - 1:
                img_array_stucki[i + 1, j] = np.clip(img_array_stucki[i + 1, j] + int(err * 8 // 42), 0, 255)
            if i < height - 1 and j < width - 1:
                img_array_stucki[i + 1, j + 1] = np.clip(img_array_stucki[i + 1, j + 1] + int(err * 4 // 42), 0, 255)
            if i < height - 1 and j < width - 2:
                img_array_stucki[i + 1, j + 2] = np.clip(img_array_stucki[i + 1, j + 2] + int(err * 2 // 42), 0, 255)
            if i < height - 2 and j > 1:
                img_array_stucki[i + 2, j - 2] = np.clip(img_array_stucki[i + 2, j - 2] + int(err // 42), 0, 255)
            if i < height - 2 and j > 0:
                img_array_stucki[i + 2, j - 1] = np.clip(img_array_stucki[i + 2, j - 1] + int(err * 2 // 42), 0, 255)
            if i < height - 2:
                img_array_stucki[i + 2, j] = np.clip(img_array_stucki[i + 2, j] + int(err * 4 // 42), 0, 255)
            if i < height - 2 and j < width - 1:
                img_array_stucki[i + 2, j + 1] = np.clip(img_array_stucki[i + 2, j + 1] + int(err * 2 // 42), 0, 255)
            if i < height - 2 and j < width - 2:
                img_array_stucki[i + 2, j + 2] = np.clip(img_array_stucki[i + 2, j + 2] + int(err // 42), 0, 255)

    # Display the resulting image using OpenCV
    cv2.imshow('Stucki Dithered Image', output_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale_to_color(image_path):
    """Transform a grayscale image to a color image using a colormap."""
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    cv2.imshow('Original Grayscale Image', gray_image)
    cv2.imshow('Color Image', color_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    original_image()
    simple_averaging()

    # Photoshop formula
    weighted_average((0.3, 0.59, 0.11), "Photoshop formula")

    #Luma formula
    weighted_average((0.2126, 0.7152, 0.0722), "Luma formula")

    #BT.601 formula
    weighted_average((0.299, 0.587, 0.114), "BT.601 formula")

    desaturation()

    decomposition_min_grayscale()

    decomposition_max_grayscale()

    single_color_channel_grayscale(0)  # Red channel
    single_color_channel_grayscale(1)  # Green channel
    single_color_channel_grayscale(2)  # Blue channel

    p = np.random.randint(8, 20)
    custom_gray_shades(p)
    floyd_steinberg_dithering(4)
    stucki_dithering(4)
    grayscale_to_color("img_reverse.png")