import os
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile

base_path = os.getcwd()

images_path = os.path.join(base_path, 'images1')
masks_path = os.path.join(base_path, 'masks1')

train_path = os.path.join(base_path, 'train1')
test_path = os.path.join(base_path, 'test1')

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

train_images_path = os.path.join(train_path, 'images1')
train_masks_path = os.path.join(train_path, 'masks1')
test_images_path = os.path.join(test_path, 'images1')
test_masks_path = os.path.join(test_path, 'masks1')

for path in [train_images_path, train_masks_path, test_images_path, test_masks_path]:
    if not os.path.exists(path):
        os.makedirs(path)

images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
masks = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]

images.sort()
masks.sort()

train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.30, random_state=42)

def copy_files(files, source, destination):
    for file in files:
        src_path = os.path.join(source, file)
        dst_path = os.path.join(destination, file)
        copyfile(src_path, dst_path)


copy_files(train_images, images_path, train_images_path)
copy_files(train_masks, masks_path, train_masks_path)
copy_files(test_images, images_path, test_images_path)
copy_files(test_masks, masks_path, test_masks_path)

print("Setul de date a fost împărțit și copiat cu succes în directoarele train și test.")
