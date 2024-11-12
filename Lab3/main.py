import os

import cv2
import numpy as np
import pandas as pd

image_path = "images/1.jpg"
image_read = cv2.imread(image_path)


##############################    Ex1     ################################


def skin_detection_rgb(img):
    binary_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    condition = (R > 95) & (G > 40) & (B > 20) & (np.abs(R - G) > 15) & (
                np.max(img, axis=2) - np.min(img, axis=2) > 15) & (R > G) & (R > B)

    binary_img[condition] = 255

    return binary_img


def skin_detection_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binary_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    H, S, V = hsv_img[:, :, 0], hsv_img[:, :, 1] / 255.0, hsv_img[:, :, 2] / 255.0
    condition = (H >= 0) & (H <= 50) & (S >= 0.23) & (S <= 0.68) & (V >= 0.35) & (V <= 1.0)

    binary_img[condition] = 255

    return binary_img


def skin_detection_ycbcr(img):
    binary_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Convert to YCbCr
    Y = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    Cb = -0.1687 * img[:, :, 2] - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 0] + 128
    Cr = 0.5 * img[:, :, 2] - 0.4187 * img[:, :, 1] - 0.0813 * img[:, :, 0] + 128

    condition = (Y > 80) & (Y <= 255) & (Cb > 85) & (Cb < 135) & (Cr > 135) & (Cr < 180)

    binary_img[condition] = 255

    return binary_img


def ex_1():
    rgb_img = skin_detection_rgb(image_read)
    cv2.imshow("Skin Detection RGB", rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv_img = skin_detection_hsv(image_read)
    cv2.imshow("Skin Detection HSV", hsv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ycbcr = skin_detection_ycbcr(image_read)
    cv2.imshow("Skin Detection YCbCr", ycbcr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


######################################   EX2    ###################################################3


def get_confusion_matrix_and_accuracy(predicted_mask, ground_truth):
    tp = np.sum((predicted_mask == 255) & (ground_truth == 255))
    fn = np.sum((predicted_mask == 0) & (ground_truth == 255))
    fp = np.sum((predicted_mask == 255) & (ground_truth == 0))
    tn = np.sum((predicted_mask == 0) & (ground_truth == 0))

    accuracy = 0
    if (tp + fn + fp + tn) > 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)

    confusion_matrix = np.array([[tp, fn], [fp, tn]])

    return confusion_matrix, accuracy


ground_truth_face_photo_path = r"Face_Dataset/Ground_Truth/GroundT_FacePhoto"
ground_truth_family_photo_path = r"Face_Dataset/Ground_Truth/GroundT_FamilyPhoto"

pratheepan_dataset_face_photo_path = r"Face_Dataset/Pratheepan_Dataset/FacePhoto"
pratheepan_dataset_family_photo_path = r"Face_Dataset/Pratheepan_Dataset/FamilyPhoto"


def ex2(ground_truth_folder, pratheepan_dataset_folder):
    accuracy_totals = {"RGB_Method": 0, "HSV_Method": 0, "YCrCb_Method": 0}
    accuracy_counts = {"RGB_Method": 0, "HSV_Method": 0, "YCrCb_Method": 0}

    for filename in os.listdir(pratheepan_dataset_folder):

        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('jpeg'):
            img_path = os.path.join(pratheepan_dataset_folder, filename)
            ground_truth_path = os.path.join(ground_truth_folder, os.path.splitext(filename)[0] + ".png")

            if not os.path.isfile(ground_truth_path):
                print(f"Ground truth file not found for {filename} at path: {ground_truth_path}")
                continue

            # read image and ground_truth
            image = cv2.imread(img_path)
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

            if image is None or ground_truth is None:
                print(f"Error reading image or ground truth for {filename}")
                continue

            ground_truth_resized = cv2.resize(ground_truth, (image.shape[1], image.shape[0]))

            for method_name, skin_method in [("RGB_Method", skin_detection_rgb), ("HSV_Method", skin_detection_hsv),
                                             ("YCrCb_Method", skin_detection_ycbcr)]:
                mask = skin_method(image)
                confusion_matrix, accuracy = get_confusion_matrix_and_accuracy(mask, ground_truth_resized)

                accuracy_totals[method_name] += accuracy
                accuracy_counts[method_name] += 1

                df_conf_matrix = pd.DataFrame(confusion_matrix, index=["Predicted Skin", "Predicted Non-Skin"],
                                              columns=["Actual Skin", "Actual Non-Skin"])
                print(f"\n{filename} ({method_name}) - Accuracy: {accuracy:.4f}")
                print(df_conf_matrix)
                print()
            print("---------------------------------------------------------------------")

    for method_name in accuracy_totals:
        if accuracy_counts[method_name] > 0:
            avg_accuracy = accuracy_totals[method_name] / accuracy_counts[method_name]
            print(f"Average accuracy for {method_name}: {avg_accuracy:.4f}")


###############################         EX3        ##################################################3


def detect_face(img_path):

    # load the image and convert to RGB
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert to HSV for skin detection
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # define skin color range in HSV
    lower_skin = np.array([0, 40, 40], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)

    #  skin mask
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # find contours of skin regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eye_positions = []

    # largest contour as the face
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        side_length = max(w, h)
        center_x, center_y = x + w // 2, y + h // 2
        x, y = max(0, center_x - side_length // 2), max(0, center_y - side_length // 2)
        face_region = image_rgb[y:y + side_length, x:x + side_length]

        # convert face region to grayscale
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        top_half = face_gray[0:int(side_length / 2.5), :]

        # threshold the top half to find dark regions (eyes)
        _, thresh = cv2.threshold(top_half, 50, 255, cv2.THRESH_BINARY_INV)

        # find contours in the threshold image
        eye_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in eye_contours:
            ex, ey, ew, eh = cv2.boundingRect(contour)
            aspect_ratio = ew / float(eh)
            if 1.2 < aspect_ratio < 3.0 and 25 < ew < 55 and ey > 30:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                axes = (ew // 2, eh // 3)
                eye_positions.append(center)
                #cv2.ellipse(image_rgb, center, axes, 0, 0, 360, (0, 255, 0), 2)

    if len(eye_positions) == 2:
        eye1, eye2 = eye_positions
        eye_distance = abs(eye1[0] - eye2[0])
        face_width = int(eye_distance * 2)
        face_height = int(face_width * 1.2)

        # readjust based on eyes
        face_x = min(eye1[0], eye2[0]) - face_width // 4
        face_y = min(eye1[1], eye2[1]) - face_height // 3

        # draw face rectangle
        cv2.rectangle(image_rgb, (face_x, face_y), (face_x + face_width, face_y + face_height), (255, 0, 0), 3)

    cv2.imshow('Refined Face and Eye Detection', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex3():
    image_paths = [
        "Face_Dataset/Pratheepan_Dataset/FacePhoto/Megan-Fox-Pretty-Face-1-1024x768.jpg",
        "Face_Dataset/Pratheepan_Dataset/FacePhoto/06Apr03Face.jpg",
        "Face_Dataset/Pratheepan_Dataset/FacePhoto/chenhao0017me9.jpg",
        "Face_Dataset/Pratheepan_Dataset/FacePhoto/920480_f520.jpg",
        "Face_Dataset/Pratheepan_Dataset/FacePhoto/Srilankan-Actress-Yamuna-Erandathi-001.jpg"
    ]

    for img_path in image_paths:
        detect_face(img_path)


if __name__ == '__main__':

    ex_1()

    print("\nFacePhoto:")
    ex2(ground_truth_face_photo_path, pratheepan_dataset_face_photo_path)
    print("\nFamilyPhoto:")
    ex2(ground_truth_family_photo_path, pratheepan_dataset_family_photo_path)

    ex3()
