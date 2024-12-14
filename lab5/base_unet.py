import os
import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from split_dataset import train_images, train_masks, test_images, test_masks


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


def base_unet_model(input_size=(256, 256, 1)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_size))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    model.add(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1, (1, 1), activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model




def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return jaccard, dice


train_images = load_images(train_images, "train/images")
train_masks = load_images(train_masks, "train/masks")
test_images = load_images(test_images, "test/images")
test_masks = load_images(test_masks, "test/masks")

print(f"Train Images Shape: {train_images.shape}")
print(f"Train Masks Shape: {train_masks.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Masks Shape: {test_masks.shape}")

model = base_unet_model()
history = model.fit(train_images, train_masks, batch_size=16, epochs=50, validation_data=(test_images, test_masks))

predictions = model.predict(test_images)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

jaccard_scores = []
dice_scores = []

for i in range(len(test_masks)):
    jaccard, dice = calculate_metrics(test_masks[i].flatten(), predictions[i].flatten())
    jaccard_scores.append(jaccard)
    dice_scores.append(dice)

mean_jaccard = np.mean(jaccard_scores)
mean_dice = np.mean(dice_scores)
pixel_accuracy = np.mean(test_masks.flatten() == predictions.flatten())

print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
print(f"Mean Jaccard Index: {mean_jaccard:.4f}")
print(f"Mean Dice Coefficient: {mean_dice:.4f}")
