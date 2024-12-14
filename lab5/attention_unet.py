import os
import cv2
import numpy as np
from keras import Model
from keras.src.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Multiply
from keras.src.optimizers import Adam
from split_dataset import train_images, train_masks, test_images, test_masks
import tensorflow as tf
from keras.src.layers import UpSampling2D
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


def attention_gate(x, gating_signal, inter_channels):
    theta_x = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(gating_signal)

    phi_g = UpSampling2D(size=(2, 2))(phi_g)

    add = Activation('relu')(theta_x + phi_g)
    psi = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(add)
    psi = Activation('sigmoid')(psi)

    return Multiply()([x, psi])



def attention_unet(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    g1 = attention_gate(c2, c3, inter_channels=64)
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, g1])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    g2 = attention_gate(c1, c4, inter_channels=32)
    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, g2])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def calculate_metrics_with_library(y_true, y_pred):

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    jaccard = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    dice = f1_score(y_true_flat, y_pred_flat, average='binary')

    return jaccard, dice



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

train_masks[train_masks > 0.5] = 1
train_masks[train_masks <= 0.5] = 0
test_masks[test_masks > 0.5] = 1
test_masks[test_masks <= 0.5] = 0


print(f"Train Images Shape: {train_images.shape}")
print(f"Train Masks Shape: {train_masks.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Masks Shape: {test_masks.shape}")

model = attention_unet()
history = model.fit(
    x=np.array(train_images),
    y=np.array(train_masks),
    batch_size=16,
    epochs=50,
    validation_data=(np.array(test_images), np.array(test_masks))
)

predictions = model.predict(test_images)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

jaccard_scores_custom = []
dice_scores_custom = []
jaccard_scores_library = []
dice_scores_library = []

for i in range(len(test_masks)):
    jaccard_custom, dice_custom = calculate_metrics(test_masks[i].flatten(), predictions[i].flatten())
    jaccard_scores_custom.append(jaccard_custom)
    dice_scores_custom.append(dice_custom)

    jaccard_lib, dice_lib = calculate_metrics_with_library(test_masks[i], predictions[i])
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