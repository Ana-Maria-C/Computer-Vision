# Lab1

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_path = "lena.tif"


def ex_2():
    image = Image.open(img_path)

    # dysplay size
    width, height = image.size
    print(f"Image size: {width}x{height}")

    # plot the image using Matplotlib
    plt.imshow(image)
    plt.title("Lena_Image")
    # hide the axis
    plt.axis('off')
    plt.show()


    # write the image

    img_to_write = cv2.imread(img_path)
    cv2.imwrite('new_lena.jpg',img_to_write)


def ex_3():

    image = cv2.imread(img_path)

    # blur the image

    # Apply blurring kernel
    kernel_blur(image)

    # Apply GaussianBlur
    gaussian_blur(image)

    #Apply MedianBlur
    median_blur(image)

    #Apply sharpening using kernel
    kernel_sharpen(image)


def kernel_blur(image):
    kernel1 = np.ones((3, 3), np.float32) / 9
    img_blur1 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

    kernel_blur2 = np.ones((5, 5), np.float32) / 25
    img_blur2 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_blur2)

    cv2.imshow('Original', image)
    cv2.imshow('Kernel Blur 1', img_blur1)
    cv2.imshow('Kernel Blur 2', img_blur2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_blur(image):
    gaussian_blur1 = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=0, sigmaY=0)
    gaussian_blur2 = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0, sigmaY=0)

    cv2.imshow('Original', image)
    cv2.imshow('Gaussian Blurred 1', gaussian_blur1)
    cv2.imshow('Gaussian Blurred 2', gaussian_blur2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median_blur(image):
    median_blur1 = cv2.medianBlur(src=image, ksize=3)
    median_blur2 = cv2.medianBlur(src=image, ksize=5)

    cv2.imshow('Median Blurred 1', median_blur1)
    cv2.imshow('Median Blurred 2', median_blur2)

    cv2.imshow('Original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kernel_sharpen(image):
    kernel1 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    sharp_img1 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

    kernel_sharpen2 = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

    sharp_img2 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_sharpen2)

    cv2.imshow('Original', image)
    cv2.imshow('Sharpened1', sharp_img1)
    cv2.imshow('Sharpened2', sharp_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_4():

    image = cv2.imread(img_path)

    # define the filter (kernel)
    kernel = np.array([[0, -2, 0],
                       [-2, 8, -2],
                       [0, -2, 0]])

    # apply the filter using cv2.filter2D()
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    plt.figure(figsize=(10, 5))

    cv2.imshow('Original', image)
    cv2.imshow('Filtered Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_5():

    image = cv2.imread(img_path)

    # get center of the image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # get the rotation matrix
    rotate_matrix1 = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
    rotate_matrix2 = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image1 = cv2.warpAffine(src=image, M=rotate_matrix1, dsize=(width, height))
    rotated_image2 = cv2.warpAffine(src=image, M=rotate_matrix2, dsize=(width, height))

    cv2.imshow('Original', image)
    cv2.imshow('Rotated image 1 ', rotated_image1)
    cv2.imshow('Rotated image 2 ', rotated_image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_6(x_start, y_start, width, length):

    image = cv2.imread(img_path)
    height_img, width_img = image.shape[:2]

    x_final = x_start + length
    y_final = y_start + width

    if x_final <= width_img and x_start >= 0 and y_start >= 0 and y_final <= height_img:
        cropped_image = image[x_start:x_final, y_start:y_final]

        cv2.imshow('Original', image)
        cv2.imshow('Cropped image ', cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex_7():

    # create blank image
    img = np.ones((500, 500, 3), dtype="uint8") * 255

    #hair
    hair_color =(0,0,180)
    cv2.rectangle(img, (78, 160), (422, 470), hair_color, -1)
    cv2.ellipse(img, (250, 160), (172, 100), 0, 0, -180, hair_color, -1)

    # draw the face
    cv2.circle(img, (250, 250), 150, (150, 192, 255), -1)
    cv2.ellipse(img, (300, 180), (20, 20), 0, 0, 180, (0, 0, 0), 4)

    # draw the eyes

    #left eye
    cv2.ellipse(img, (200, 180), (20, 20), 0, 0, 180, (0, 0, 0), 4)

    start_point = (180, 180)
    end_point = (220, 180)
    color = (0, 0, 0)
    thickness = 4
    cv2.line(img, start_point, end_point, color, thickness)
    cv2.circle(img, (200, 190), 5, (250,150,50), -1)

    #eyelashes
    cv2.line(img, (175, 165), (180, 180), color, 4)
    cv2.line(img, (185, 165), (185, 180), color, 3)
    cv2.line(img, (192, 165), (192, 180), color, 3)
    cv2.line(img, (199, 165), (199, 180), color, 3)
    cv2.line(img, (207, 165), (207, 180), color, 3)
    cv2.line(img, (215, 165), (215, 180), color, 3)
    cv2.line(img, (225, 165), (220, 180), color, 3)

    #right eye
    cv2.ellipse(img, (300, 180), (20, 20), 0, 0, 180, (0, 0, 0), 4)

    start_point = (280, 180)
    end_point = (320, 180)
    color = (0, 0, 0)
    thickness = 4

    cv2.line(img, start_point, end_point, color, thickness)

    cv2.circle(img, (300, 190), 5, (250,150,50), -1)

    # eyelashes
    cv2.line(img, (275, 165), (280, 180), color, 4)
    cv2.line(img, (285, 165), (285, 180), color, 3)
    cv2.line(img, (292, 165), (292, 180), color, 3)
    cv2.line(img, (299, 165), (299, 180), color, 3)
    cv2.line(img, (307, 165), (307, 180), color, 3)
    cv2.line(img, (315, 165), (315, 180), color, 3)
    cv2.line(img, (325, 165), (320, 180), color, 3)

    # mouth
    cv2.ellipse(img, (250, 300), (50, 35), 0, 0, 180, (0, 0, 255), -1)
    cv2.ellipse(img, (225, 300), (25, 20), 0, 0, -180, (0, 0, 255), -1)
    cv2.ellipse(img, (275, 300), (25, 20), 0, 0, -180, (0, 0, 255), -1)
    cv2.line(img, (200, 300), (300, 300), (0, 0, 200), 2)

    # hair
    cv2.ellipse(img, (110, 100), (140, 60), 0, 0, 90, hair_color, -1)
    cv2.ellipse(img, (250, 110), (140, 50), 0, 0, -180, hair_color, -1)
    cv2.ellipse(img, (250, 100), (120, 50), 0, 0, -180, hair_color, -1)
    cv2.ellipse(img, (380, 100), (140, 60), 0, -180, -270, hair_color, -1)

    #cheeks
    #cv2.circle(img, (350, 250), 30, (100, 100, 255), -1)
    #cv2.circle(img, (150, 250), 30, (100, 100, 255), -1)
    draw_heart(img, (350,250), 50, 50, (100, 100, 255))
    draw_heart(img, (150, 250), 50, 50, (100, 100, 255))

    cv2.circle(img, (310, 280), 2, (0, 0, 0), -1)

    # display the emoji image
    cv2.imshow("Emoji", img)
    cv2.imwrite("Constantin_Ana-Maria_emoji.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_heart(img, center, width, height, color):

    center_left = (center[0] - width // 4, center[1] - height // 4)  # left
    center_right = (center[0] + width // 4, center[1] - height // 4)  # right
    axes = (width // 4, height // 4)

    # left ellipse
    cv2.ellipse(img, center_left, axes, 0, 0, -180, color, -1)

    # right ellipse
    cv2.ellipse(img, center_right, axes, 0, 0, -180, color, -1)

    # triangle for the bottom of the heart
    points = np.array([[center[0] - width // 2, center[1] - height // 4],
                       [center[0] + width // 2, center[1] - height // 4],
                       [center[0], center[1] + height // 2]], np.int32)
    cv2.fillPoly(img, [points], color)


if __name__ == '__main__':
    ex_2()
    ex_3()
    ex_4()
    ex_5()
    ex_6(200, 200,200,100)
    ex_7()