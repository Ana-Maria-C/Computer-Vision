import cv2

from ex1 import OCRProcessor
from ex2 import add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
from ex3 import rotate_image, shear_image
from ex4 import resize_with_aspect_ratio, resize_alter_aspect_ratio
from ex5 import apply_gaussian_blur, apply_average_blur
from ex6 import apply_sharpening, apply_erosion, apply_dilation, apply_closing,apply_opening,apply_simple_threshold,apply_adaptive_threshold


def main():
    image_path = "images/bon_fiscal.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_path = "ground_truth/test_img3.txt"

    if image is None:
        raise ValueError("Image not found. Check the file path provided.")

    ############################## ex1 ################################

    ocr_processor = OCRProcessor()

    ocr_result = ocr_processor.perform_ocr(image)
    #print("OCR Result:")
    #print(ocr_result)

    accuracy = ocr_processor.get_accuracy(ground_truth_path)
    print("EX1")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("---------------------------------------------------------\n")

    ######################################### ex2 #######################

    gaussian_noisy = add_gaussian_noise(image)
    salt_pepper_noisy = add_salt_and_pepper_noise(image)
    speckle_noisy = add_speckle_noise(image)

    # Display the results
    titles = ['Original', 'Gaussian Noise', 'Salt & Pepper Noise', 'Speckle Noise']
    images = [image, gaussian_noisy, salt_pepper_noisy, speckle_noisy]

    print("EX2")
    for img, title in zip(images, titles):
        ocr_result = ocr_processor.perform_ocr(img)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"Accuracy for {title} : {acc * 100:.2f}%")
        print()
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("----------------------------------------------------------")

    #################################### ex3 #################################

    print("EX3")
    for angle in [15, 45, 75]:
        rotated = rotate_image(image, angle)
        ocr_result = ocr_processor.perform_ocr(rotated)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"Accuracy for Rotated {angle}° : {acc * 100:.2f}%")
        print()
        cv2.imshow(f"Rotated {angle}°", rotated)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Test with different shearing factors
    for shear_x in [0.1, 0.3, 0.5]:
        sheared = shear_image(image, shear_factor_x=shear_x)
        ocr_result = ocr_processor.perform_ocr(sheared)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"Accuracy for Horizontally Sheared factor={shear_x} : {acc * 100:.2f}%")
        print()
        cv2.imshow(f"Horizontally Sheared factor={shear_x})", sheared)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    for shear_y in [0.1, 0.3, 0.5]:
        sheared = shear_image(image, shear_factor_y=shear_y)
        ocr_result = ocr_processor.perform_ocr(sheared)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"Accuracy for Vertically Sheared (factor={shear_y} : {acc * 100:.2f}%")
        print()
        cv2.imshow(f"Vertically Sheared (factor={shear_y})", sheared)
        cv2.waitKey(0)
    print("----------------------------------------------------------")

    cv2.destroyAllWindows()


    #################################### ex4 #################################

    print("EX4")
    resized_half = resize_with_aspect_ratio(image, 0.5)
    resized_double = resize_with_aspect_ratio(image, 2.0)

    resized_stretched = resize_alter_aspect_ratio(image, 300, 100)
    resized_compressed = resize_alter_aspect_ratio(image, 100, 300)

    images = [image, resized_half, resized_double, resized_stretched, resized_compressed]
    titles = ["Original Image", "Resized (Half Size)", "Resized (Double Size)", "Stretched (300x100)",
              "Compressed (100x300)"]

    for img, title in zip(images, titles):
        ocr_result = ocr_processor.perform_ocr(img)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"{title} : {acc * 100:.2f}%")
        print()
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("------------------------------------------------------------------------- ")

    #################################### ex5 #################################
    print("EX5")
    average_blur_3 = apply_average_blur(image, 3)  # 3x3 kernel
    average_blur_5 = apply_average_blur(image, 5)  # 5x5 kernel
    average_blur_7 = apply_average_blur(image, 7)  # 7x7 kernel

    gaussian_blur_3_1 = apply_gaussian_blur(image, 3, 1)  # 3x3 kernel, sigma=1
    gaussian_blur_5_2 = apply_gaussian_blur(image, 5, 2)  # 5x5 kernel, sigma=2
    gaussian_blur_7_3 = apply_gaussian_blur(image, 7, 3)  # 7x7 kernel, sigma=3

    images = [image, average_blur_3, average_blur_5, average_blur_7, gaussian_blur_3_1, gaussian_blur_5_2,
              gaussian_blur_7_3]
    titles = ["Original Image", "Average Blur (3x3)", "Average Blur (5x5)", "Average Blur (7x7)",
              "Gaussian Blur (3x3, sigma=1)", "Gaussian Blur (5x5, sigma=2)", "Gaussian Blur (7x7, sigma=3)"]

    for img, title in zip(images, titles):
        ocr_result = ocr_processor.perform_ocr(img)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"{title} : {acc * 100:.2f}%")
        print()
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("------------------------------------------------------------------------- ")

    #################################### ex5 #################################

    print("EX6")
    sharpened_image = apply_sharpening(image)
    erosion_image = apply_erosion(image, 5)
    dilation_image = apply_dilation(image, 5)
    opening_image = apply_opening(image, 5)
    closing_image = apply_closing(image, 5)
    simple_threshold_image = apply_simple_threshold(image, 127)
    adaptive_threshold_image = apply_adaptive_threshold(image)

    images = [image, sharpened_image, erosion_image, dilation_image, opening_image, closing_image,
              simple_threshold_image, adaptive_threshold_image]
    titles = ["Original Image", "Sharpened Image", "Erosion", "Dilation", "Opening", "Closing", "Simple Threshold",
              "Adaptive Threshold"]

    for img, title in zip(images, titles):
        ocr_result = ocr_processor.perform_ocr(img)
        acc = ocr_processor.get_accuracy(ground_truth_path)
        print(f"{title} : {acc * 100:.2f}%")
        print()
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()