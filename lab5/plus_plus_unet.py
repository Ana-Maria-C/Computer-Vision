import os
import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import Model
from keras.src.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization , Input, concatenate, UpSampling2D
from keras.src.optimizers import Adam
from split_dataset import train_images, train_masks, test_images, test_masks
from sklearn.metrics import jaccard_score, f1_score


def load_images(file_list, base_path, target_size=(128, 128)):
    images = []
    for file in file_list:
        img_path = os.path.join(base_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping missing file: {file}")
            continue
        img = cv2.resize(img, target_size)
        img = img / 255.0
        images.append(img)
    return np.array(images, dtype=np.float32).reshape(-1, target_size[0], target_size[1], 1)


def unetplusplus(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    c6_resized = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3, c6_resized])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    c7_resized = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2, c7_resized])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    c8_resized = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1, c8_resized])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return jaccard, dice


def calculate_metrics_with_library(y_true, y_pred):

    y_true_binary = (y_true > 0.5).astype(int).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    jaccard = jaccard_score(y_true_binary, y_pred_binary, average='binary')
    dice = f1_score(y_true_binary, y_pred_binary, average='binary')

    return jaccard, dice


train_images = load_images(train_images, "train/images")
train_masks = load_images(train_masks, "train/masks")
test_images = load_images(test_images, "test/images")
test_masks = load_images(test_masks, "test/masks")

print(f"Train Images Shape: {train_images.shape}")
print(f"Train Masks Shape: {train_masks.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Masks Shape: {test_masks.shape}")

model = unetplusplus()
history = model.fit(train_images, train_masks, batch_size=16, epochs=50, validation_data=(test_images, test_masks))

predictions = model.predict(test_images)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

jaccard_scores_custom = []
dice_scores_custom = []
jaccard_scores_library = []
dice_scores_library = []

for i in range(len(test_masks)):
    y_true_binary = (test_masks[i] > 0.5).astype(int)
    y_pred_binary = (predictions[i] > 0.5).astype(int)

    jaccard_custom, dice_custom = calculate_metrics(y_true_binary.flatten(), y_pred_binary.flatten())
    jaccard_scores_custom.append(jaccard_custom)
    dice_scores_custom.append(dice_custom)

    jaccard_lib, dice_lib = calculate_metrics_with_library(y_true_binary, y_pred_binary)
    jaccard_scores_library.append(jaccard_lib)
    dice_scores_library.append(dice_lib)

mean_jaccard_custom = np.mean(jaccard_scores_custom)
mean_dice_custom = np.mean(dice_scores_custom)
mean_jaccard_library = np.mean(jaccard_scores_library)
mean_dice_library = np.mean(dice_scores_library)
pixel_accuracy = np.mean(test_masks.flatten() == predictions.flatten())

print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
print(f"Mean Jaccard Index (Custom): {mean_jaccard_custom:.4f}")
print(f"Mean Dice Coefficient (Custom): {mean_dice_custom:.4f}")
print(f"Mean Jaccard Index (Library): {mean_jaccard_library:.4f}")
print(f"Mean Dice Coefficient (Library): {mean_dice_library:.4f}")

